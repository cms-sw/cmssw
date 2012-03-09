
#include "RecoEcal/EgammaClusterAlgos/interface/Multi5x5ClusterAlgo.h"

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "RecoEcal/EgammaCoreTools/interface/ClusterEtLess.h"

#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Return a vector of clusters from a collection of EcalRecHits:
//
std::vector<reco::BasicCluster> Multi5x5ClusterAlgo::makeClusters(
        const EcalRecHitCollection* hits,
        const CaloSubdetectorGeometry *geometry_p,
        const CaloSubdetectorTopology *topology_p,
        const CaloSubdetectorGeometry *geometryES_p,
        reco::CaloID::Detectors detector,
        bool regional,
        const std::vector<EcalEtaPhiRegion>& regions
        )
{
    seeds.clear();
    used_s.clear();
    canSeed_s.clear();
    clusters_v.clear();

    recHits_ = hits;

    double threshold = 0;
    std::string ecalPart_string;
    detector_ = reco::CaloID::DET_NONE;
    if (detector == reco::CaloID::DET_ECAL_ENDCAP) 
    {
        detector_ = reco::CaloID::DET_ECAL_ENDCAP;
        threshold = ecalEndcapSeedThreshold;
        ecalPart_string = "EndCap";
    }
    if (detector == reco::CaloID::DET_ECAL_BARREL) 
    {
        detector_ = reco::CaloID::DET_ECAL_BARREL;
        threshold = ecalBarrelSeedThreshold;
        ecalPart_string = "Barrel";
    }

    LogTrace("EcalClusters") << "-------------------------------------------------------------";
    LogTrace("EcalClusters") << "Island algorithm invoked for ECAL" << ecalPart_string ;
    LogTrace("EcalClusters") << "Looking for seeds, energy threshold used = " << threshold << " GeV";


    int nregions=0;
    if(regional) nregions=regions.size();

    if(!regional || nregions) {

        EcalRecHitCollection::const_iterator it;
        for(it = hits->begin(); it != hits->end(); it++)
        {
            double energy = it->energy();
            if (energy < threshold) continue; // need to check to see if this line is useful!

            const CaloCellGeometry *thisCell = geometry_p->getGeometry(it->id());
            GlobalPoint position = thisCell->getPosition();

            // Require that RecHit is within clustering region in case
            // of regional reconstruction
            bool withinRegion = false;
            if (regional) {
                std::vector<EcalEtaPhiRegion>::const_iterator region;
                for (region=regions.begin(); region!=regions.end(); region++) {
                    if (region->inRegion(position)) {
                        withinRegion =  true;
                        break;
                    }
                }
            }

            if (!regional || withinRegion) {
                float ET = it->energy() * sin(position.theta());
                if (ET > threshold) seeds.push_back(*it);
            }
        }

    }

    sort(seeds.begin(), seeds.end(), EcalRecHitLess());

    LogTrace("EcalClusters") << "Total number of seeds found in event = " << seeds.size();


    mainSearch(hits, geometry_p, topology_p, geometryES_p);
    sort(clusters_v.rbegin(), clusters_v.rend(), ClusterEtLess());

    LogTrace("EcalClusters") << "---------- end of main search. clusters have been sorted ----";


    return clusters_v;

}

// Search for clusters
//
void Multi5x5ClusterAlgo::mainSearch(const EcalRecHitCollection* hits,
        const CaloSubdetectorGeometry *geometry_p,
        const CaloSubdetectorTopology *topology_p,
        const CaloSubdetectorGeometry *geometryES_p
        )
{

    LogTrace("EcalClusters") << "Building clusters............";

    // Loop over seeds:
    std::vector<EcalRecHit>::iterator it;
    for (it = seeds.begin(); it != seeds.end(); it++)
    {

        // check if this crystal is able to seed
        // (event though it is already used)
        bool usedButCanSeed = false;
        if (canSeed_s.find(it->id()) != canSeed_s.end()) usedButCanSeed = true;

        // avoid seeding for anomalous channels (recoFlag based)
        uint32_t rhFlag = (*it).recoFlag();
        std::vector<int>::const_iterator vit = std::find( v_chstatus_.begin(), v_chstatus_.end(), rhFlag );
        if ( vit != v_chstatus_.end() ) continue; // the recHit has to be excluded from seeding

        // make sure the current seed does not belong to a cluster already.
        if ((used_s.find(it->id()) != used_s.end()) && (usedButCanSeed == false))
        {
            if (it == seeds.begin())
            {
                LogTrace("EcalClusters") << "##############################################################" ;
                LogTrace("EcalClusters") << "DEBUG ALERT: Highest energy seed already belongs to a cluster!";
                LogTrace("EcalClusters") << "##############################################################";

            }

            // seed crystal is used or is used and cannot seed a cluster
            // so continue to the next seed crystal...
            continue;
        }

        // clear the vector of hits in current cluster
        current_v.clear();

        // Create a navigator at the seed and get seed
        // energy
        CaloNavigator<DetId> navigator(it->id(), topology_p);
        DetId seedId = navigator.pos();
        EcalRecHitCollection::const_iterator seedIt = hits->find(seedId);
        navigator.setHome(seedId);

        // Is the seed a local maximum?
        bool localMaxima = checkMaxima(navigator, hits);

        if (localMaxima)
        {
            // build the 5x5 taking care over which crystals
            // can seed new clusters and which can't
            prepareCluster(navigator, hits, geometry_p);
        }

        // If some crystals in the current vector then 
        // make them into a cluster 
        if (current_v.size() > 0) 
        {
            makeCluster(hits, geometry_p, geometryES_p, seedIt, usedButCanSeed);
        }

    }  // End loop on seed crystals

}

void Multi5x5ClusterAlgo::makeCluster(const EcalRecHitCollection* hits,
        const CaloSubdetectorGeometry *geometry,
        const CaloSubdetectorGeometry *geometryES,
        const EcalRecHitCollection::const_iterator &seedIt,
        bool seedOutside)
{

    double energy = 0;
    //double chi2   = 0;
    reco::CaloID caloID;
    Point position;
    position = posCalculator_.Calculate_Location(current_v, hits,geometry, geometryES);

    std::vector<std::pair<DetId, float> >::iterator it;
    for (it = current_v.begin(); it != current_v.end(); it++)
    {
        EcalRecHitCollection::const_iterator itt = hits->find( (*it).first );
        EcalRecHit hit_p = *itt;
        energy += hit_p.energy();
        //chi2 += 0;
        if ( (*it).first.subdetId() == EcalBarrel ) {
            caloID = reco::CaloID::DET_ECAL_BARREL;
        } else {
            caloID = reco::CaloID::DET_ECAL_ENDCAP;
        }

    }
    //chi2 /= energy;

    LogTrace("EcalClusters") << "******** NEW CLUSTER ********";
    LogTrace("EcalClusters") << "No. of crystals = " << current_v.size();
    LogTrace("EcalClusters") << "     Energy     = " << energy ;
    LogTrace("EcalClusters") << "     Phi        = " << position.phi();
    LogTrace("EcalClusters") << "     Eta " << position.eta();
    LogTrace("EcalClusters") << "*****************************";  


    // to be a valid cluster the cluster energy
    // must be at least the seed energy
    double seedEnergy = seedIt->energy();
    if ((seedOutside && energy>=0) || (!seedOutside && energy >= seedEnergy)) 
    {
        clusters_v.push_back(reco::BasicCluster(energy, position, reco::CaloID(detector_), current_v, reco::CaloCluster::multi5x5, seedIt->id()));

    // if no valid cluster was built,
    // then free up these crystals to be used in the next...
    } else {
        std::vector<std::pair<DetId, float> >::iterator iter;
        for (iter = current_v.begin(); iter != current_v.end(); iter++)
        {
            used_s.erase(iter->first);
        } //for(iter)
    } //else

}

bool Multi5x5ClusterAlgo::checkMaxima(CaloNavigator<DetId> &navigator,
        const EcalRecHitCollection *hits)
{

    bool maxima = true;
    EcalRecHitCollection::const_iterator thisHit;
    EcalRecHitCollection::const_iterator seedHit = hits->find(navigator.pos());
    double seedEnergy = seedHit->energy();

    std::vector<DetId> swissCrossVec;
    swissCrossVec.clear();

    swissCrossVec.push_back(navigator.west());
    navigator.home();
    swissCrossVec.push_back(navigator.east());
    navigator.home();
    swissCrossVec.push_back(navigator.north());
    navigator.home();
    swissCrossVec.push_back(navigator.south());
    navigator.home();

    std::vector<DetId>::const_iterator detItr;
    for (unsigned int i = 0; i < swissCrossVec.size(); ++i)
    {

        // look for this hit
        thisHit = recHits_->find(swissCrossVec[i]);

        // continue if this hit was not found
        if  ((swissCrossVec[i] == DetId(0)) || thisHit == recHits_->end()) continue; 

        // the recHit has to be skipped in the local maximum search if it was found
        // in the map of channels to be excluded 
        uint32_t rhFlag = thisHit->recoFlag();
        std::vector<int>::const_iterator vit = std::find(v_chstatus_.begin(), v_chstatus_.end(), rhFlag);
        if (vit != v_chstatus_.end()) continue;

        // if this crystal has more energy than the seed then we do 
        // not have a local maxima
        if (thisHit->energy() > seedEnergy)
        {
            maxima = false;
            break;
        }
    }

    return maxima;

}

void Multi5x5ClusterAlgo::prepareCluster(CaloNavigator<DetId> &navigator, 
        const EcalRecHitCollection *hits, 
        const CaloSubdetectorGeometry *geometry)
{

    DetId thisDet;
    std::set<DetId>::iterator setItr;

    // now add the 5x5 taking care to mark the edges
    // as able to seed and where overlapping in the central
    // region with crystals that were previously able to seed
    // change their status so they are not able to seed
    //std::cout << std::endl;
    for (int dx = -2; dx < 3; ++dx)
    {
        for (int dy = -2; dy < 3; ++ dy)
        {

            // navigate in free steps forming
            // a full 5x5
            thisDet = navigator.offsetBy(dx, dy);
            navigator.home();

            // add the current crystal
            //std::cout << "adding " << dx << ", " << dy << std::endl;
            addCrystal(thisDet);

            // now consider if we are in an edge (outer 16)
            // or central (inner 9) region
            if ((abs(dx) > 1) || (abs(dy) > 1))
            {    
                // this is an "edge" so should be allowed to seed
                // provided it is not already used
                //std::cout << "   setting can seed" << std::endl;
                canSeed_s.insert(thisDet);
            }  // end if "edge"
            else 
            {
                // or else we are in the central 3x3
                // and must remove any of these crystals from the canSeed set
                setItr = canSeed_s.find(thisDet);
                if (setItr != canSeed_s.end())
                {
                    //std::cout << "   unsetting can seed" << std::endl;
                    canSeed_s.erase(setItr);
                }
            }  // end if "centre"


        } // end loop on dy

    } // end loop on dx

    //std::cout << "*** " << std::endl;
    //std::cout << " current_v contains " << current_v.size() << std::endl;
    //std::cout << "*** " << std::endl;
}


void Multi5x5ClusterAlgo::addCrystal(const DetId &det)
{   

    EcalRecHitCollection::const_iterator thisIt =  recHits_->find(det);
    if ((thisIt != recHits_->end()) && (thisIt->id() != DetId(0)))
    { 
        if ((used_s.find(thisIt->id()) == used_s.end())) 
        {
            //std::cout << "   ... this is a good crystal and will be added" << std::endl;
            current_v.push_back( std::pair<DetId, float>(det, 1.) ); // by default hit energy fractions are set at 1.
            used_s.insert(det);
        }
    } 

}

