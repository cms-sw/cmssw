// -*- C++ -*-
//
// Package: L1CaloTrigger
// Class: L1CaloTauProducer
//
/**\class L1CaloTauProducer L1CaloTauProducer.cc

Description: 
Beginning with HCAL TPs, create HCAL jet, then
take L1EG crystal clusters from L1EGammaCrystalsProducer.cc
and clusters them within fixed number of trigger towers

Implementation:
[Notes on implementation]
*/
//
// Original Author: Tyler Ruggles
// Created: Tue Aug 29 2018
// $Id$
//
//


#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include <iostream>

#include "DataFormats/Phase2L1CaloTrig/interface/L1EGCrystalCluster.h"
#include "DataFormats/Phase2L1CaloTrig/interface/L1CaloJet.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"

// For pT calibrations
#include "TF1.h"

// HCAL TPs
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"

// Run2/PhaseI EG object:
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"

#include <bitset>

class L1CaloTauProducer : public edm::EDProducer {
    public:
        explicit L1CaloTauProducer(const edm::ParameterSet&);

    private:
        virtual void produce(edm::Event&, const edm::EventSetup&);
        //bool cluster_passes_base_cuts(float &cluster_pt, float &cluster_eta, float &iso, float &e2x5, float &e5x5) const;
        int ecalXtal_diPhi( int &iPhi_1, int &iPhi_2 ) const;
        int hcalTower_diPhi( int &iPhi_1, int &iPhi_2 ) const;

        //double EtminForStore;
        //double EcalTpEtMin;
        double EtMinForSeedHit;
        bool debug;
        edm::EDGetTokenT< edm::SortedCollection<HcalTriggerPrimitiveDigi> > hcalTPToken_;

        //edm::EDGetTokenT<BXVector<l1t::EGamma>> crystalClustersToken_;
        //edm::Handle<BXVector<l1t::EGamma>> crystalClustersHandle;
        //BXVector<l1t::EGamma> crystalClusters;
        edm::EDGetTokenT<l1slhc::L1EGCrystalClusterCollection> crystalClustersToken_;
        edm::Handle<l1slhc::L1EGCrystalClusterCollection> crystalClustersHandle;
        l1slhc::L1EGCrystalClusterCollection crystalClusters;

        edm::ESHandle<CaloGeometry> caloGeometry_;
        const CaloSubdetectorGeometry * hbGeometry;
        edm::ESHandle<HcalTopology> hbTopology;
        const HcalTopology * hcTopology_;

        // Fit function to scale L1EG Pt to align with electron gen pT
        TF1 ptAdjustFunc = TF1("ptAdjustFunc", "([0] + [1]*TMath::Exp(-[2]*x)) * ([3] + [4]*TMath::Exp(-[5]*x))");


        class simpleL1obj
        {
            public:
                bool stale=false; // Hits become stale once used in clustering algorithm to prevent overlap in clusters
                bool passesStandaloneWP = false; // Store whether any of the WPs are passed
                bool passesTrkMatchWP = false; // Store whether any of the WPs are passed
                reco::Candidate::PolarLorentzVector p4;
                int iEta;
                int iPhi;

                void SetP4( double pt, double eta, double phi, double mass )
                {
                    this->p4.SetPt( pt );
                    this->p4.SetEta( eta );
                    this->p4.SetPhi( phi );
                    this->p4.SetM( mass );
                }
                inline float pt() const{return p4.pt();};
                inline float eta() const{return p4.eta();};
                inline float phi() const{return p4.phi();};
                inline float M() const{return p4.M();};
                inline reco::Candidate::PolarLorentzVector GetP4() const{return p4;};
        };
                


        class SimpleCaloHit
        {
            public:
                EBDetId id;
                int hcal_iEta = -99;
                int hcal_iPhi = -99;
                GlobalVector position; // As opposed to GlobalPoint, so we can add them (for weighted average)
                float energy=0.;
                bool stale=false; // Hits become stale once used in clustering algorithm to prevent overlap in clusters
                bool isEndcapHit=false; // If using endcap, we won't be using integer crystal indices
                
            // tool functions
                inline float pt() const{return (position.mag2()>0) ? energy*sin(position.theta()) : 0.;};
                inline float deta(SimpleCaloHit& other) const{return position.eta() - other.position.eta();};
                int dieta(SimpleCaloHit& other) const
                {
                    if ( isEndcapHit || other.isEndcapHit ) return 9999; // We shouldn't compare integer indices in endcap, the map is not linear
                    // int indices do not contain zero
                    // Logic from EBDetId::distanceEta() without the abs()
                    if (id.ieta() * other.id.ieta() > 0)
                        return id.ieta()-other.id.ieta();
                    return id.ieta()-other.id.ieta()-1;
                };
                inline float dphi(SimpleCaloHit& other) const{return reco::deltaPhi(static_cast<float>(position.phi()), static_cast<float>(other.position.phi()));};
                int diphi(SimpleCaloHit& other) const
                {
                    if ( isEndcapHit || other.isEndcapHit ) return 9999; // We shouldn't compare integer indices in endcap, the map is not linear
                    // Logic from EBDetId::distancePhi() without the abs()
                    int PI = 180;
                    int result = id.iphi() - other.id.iphi();
                    while (result > PI) result -= 2*PI;
                    while (result <= -PI) result += 2*PI;
                    return result;
                };
                float distanceTo(SimpleCaloHit& other) const
                {
                    // Treat position as a point, measure 3D distance
                    // This is used for endcap hits, where we don't have a rectangular mapping
                    return (position-other.position).mag();
                };
                bool operator==(SimpleCaloHit& other) const
                {
                    if ( id == other.id &&
                          position == other.position &&
                          energy == other.energy &&
                          isEndcapHit == other.isEndcapHit
                        ) return true;
                        
                    return false;
                };
        };
};

L1CaloTauProducer::L1CaloTauProducer(const edm::ParameterSet& iConfig) :
    //EtminForStore(iConfig.getParameter<double>("EtminForStore")),
    //EcalTpEtMin(iConfig.getUntrackedParameter<double>("EcalTpEtMin", 0.5)), // Default to 500 MeV
    EtMinForSeedHit(iConfig.getUntrackedParameter<double>("EtMinForSeedHit", 2.5)), // Default to 2.5 GeV
    debug(iConfig.getUntrackedParameter<bool>("debug", false)),
    hcalTPToken_(consumes< edm::SortedCollection<HcalTriggerPrimitiveDigi> >(iConfig.getParameter<edm::InputTag>("hcalTP"))),
    //crystalClustersToken_(consumes<BXVector<l1t::EGamma>>(iConfig.getParameter<edm::InputTag>("L1CrystalClustersInputTag")))
    crystalClustersToken_(consumes<l1slhc::L1EGCrystalClusterCollection>(iConfig.getParameter<edm::InputTag>("L1CrystalClustersInputTag")))

{
    produces<l1slhc::L1CaloJetsCollection>("L1CaloTausNoCuts");
    produces<l1slhc::L1CaloJetsCollection>("L1CaloTausWithCuts");
    //produces<l1extra::L1JetParticleCollection>("L1CaloClusterCollectionWithCuts");
    produces< BXVector<l1t::Tau> >("L1CaloClusterCollectionBXVWithCuts");


    // Fit parameters measured on 11 Aug 2018, using 500 MeV threshold for ECAL TPs
    // working in CMSSW 10_1_7
    // Adjustments to be applied to reco cluster pt
    // L1EG cut working points are still a function of non-calibrated pT
    // First order corrections
    ptAdjustFunc.SetParameter( 0, 1.06 );
    ptAdjustFunc.SetParameter( 1, 0.273 );
    ptAdjustFunc.SetParameter( 2, 0.0411 );
    // Residuals
    ptAdjustFunc.SetParameter( 3, 1.00 );
    ptAdjustFunc.SetParameter( 4, 0.567 );
    ptAdjustFunc.SetParameter( 5, 0.288 );
}

void L1CaloTauProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

    // Output collections
    std::unique_ptr<l1slhc::L1CaloJetsCollection> L1CaloTausNoCuts (new l1slhc::L1CaloJetsCollection );
    std::unique_ptr<l1slhc::L1CaloJetsCollection> L1CaloTausWithCuts( new l1slhc::L1CaloJetsCollection );
    //std::unique_ptr<l1extra::L1JetParticleCollection> L1CaloClusterCollectionWithCuts( new l1extra::L1JetParticleCollection );
    std::unique_ptr<BXVector<l1t::Tau>> L1CaloClusterCollectionBXVWithCuts(new l1t::TauBxCollection);



    // Get calo geometry info split by subdetector
    iSetup.get<CaloGeometryRecord>().get(caloGeometry_);
    hbGeometry = caloGeometry_->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
    iSetup.get<HcalRecNumberingRecord>().get(hbTopology);
    hcTopology_ = hbTopology.product();
    HcalTrigTowerGeometry theTrigTowerGeometry(hcTopology_);
    iEvent.getByToken(crystalClustersToken_,crystalClustersHandle);
    crystalClusters = (*crystalClustersHandle.product());


    
    // Load HCAL TPs which have ET > 0 into hcalhits
    // This section is directly from L1EGammaCrystalsProducer.cc
    std::vector<SimpleCaloHit> hcalhits;
    
    edm::Handle< edm::SortedCollection<HcalTriggerPrimitiveDigi> > hbhecoll;
    iEvent.getByToken(hcalTPToken_,hbhecoll);
    for (auto& hit : *hbhecoll.product())
    {

        // SOI_compressedEt() Compressed ET, integer representing increments of 500 MeV
        // Cut requires 500 MeV TP
        if ( hit.SOI_compressedEt() == 0 ) continue; // SOI_compressedEt() Compressed ET for the "Sample of Interest"
        // Need to use proper decompression here https://github.com/cms-sw/cmssw/blob/CMSSW_9_0_X/L1Trigger/L1TCaloLayer1/src/L1TCaloLayer1FetchLUTs.cc#L97-L114

        //std::cout << "  -- HCAL TP: " << hit.SOI_compressedEt() << std::endl;
        //auto cell = geometryHelper.getHcalGeometry()->getGeometry(hit.id());
        //std::cout << "Hit ID: " << hit.id() << std::endl;

        // Check if the detId is in the current geometry setup
        // so that L1EG doesn't crash
        if (!(hcTopology_->validHT(hit.id()))) {
          std::cout << " -- Hcal hit DetID not present in HCAL Geom: " << hit.id() << std::endl;
          continue;
        }

        // Get the detId associated with the HCAL TP
        // if no detIds associated, skip
        std::vector<HcalDetId> hcId = theTrigTowerGeometry.detIds(hit.id());
        if (hcId.size() == 0) {
          std::cout << "Cannot find any HCalDetId corresponding to " << hit.id() << std::endl;
          continue;
        }

        // Skip HCAL TPs which don't have HB detIds
        if (hcId[0].subdetId() > 1) continue;

        // Find the average position of all HB detIds
        GlobalVector hcal_tp_position = GlobalVector(0., 0., 0.);
        for (auto &hcId_i : hcId) {
          if (hcId_i.subdetId() > 1) continue;
          auto cell = hbGeometry->getGeometry(hcId_i);
          if (cell == 0) continue;
          GlobalVector tmpVector = GlobalVector(cell->getPosition().x(), cell->getPosition().y(), cell->getPosition().z());
          if ( debug ) std::cout << " ---- " << hcId_i << "  subD: " << hcId_i.subdetId() << " : (eta,phi,z), (" << tmpVector.eta() << ", " << tmpVector.phi() << ", " << tmpVector.z() << "), (iEta,iPhi) (" << hit.id().ieta() << ", " << hit.id().iphi() << ")" << std::endl;
          hcal_tp_position = tmpVector;
          break;
        }

        SimpleCaloHit hhit;
        hhit.id = hit.id();
        hhit.hcal_iEta = hit.id().ieta();
        hhit.hcal_iPhi = hit.id().iphi();
        hhit.position = hcal_tp_position;
        float et = hit.SOI_compressedEt() / 2.;
        hhit.energy = et / sin(hhit.position.theta());
        hcalhits.push_back(hhit);

        //std::cout << "HCAL TP Position (x,y,z): " << hcal_tp_position << ", TP ET : " << hhit.energy << std::endl;
        //std::cout << " - position eta,phi " << hhit.position.eta() << ":" << hhit.position.phi() << std::endl;
    }

    // Sort the HCAL TPs
    std::sort(begin(hcalhits), end(hcalhits), [](const SimpleCaloHit& a,
            SimpleCaloHit& b){return a.pt() > b.pt();});



    /**************************************************************************
 * Begin with making HCAL-based CaloJets in 11x11 grid.
 * For reference, Run-I used 12x12 grid, 11x11 allows us to conveniently have a
 * central seed tower.
 ******************************************************************************/

    // Experimental parameters, don't want to bother with hardcoding them in data format
    std::map<std::string, float> params;

    // Create hcalJetCluster within 11x11 of highest ET seed HCAL TT.
    // 11 trigger towers gives diameter 0.957
    std::vector< reco::Candidate::PolarLorentzVector > jetClusters;
    std::vector< reco::Candidate::PolarLorentzVector > hcalJetClusters;
    std::vector< std::vector< float > > jetClustersHcalInfo;

    // Count the number of unused HCAL TPs so we can stop while loop after done.
    // Clustering can also stop once there are no seed hits >= EtMinForSeedHit
    int num_unused_hits = hcalhits.size();
    bool hcalJetClusteringFinished = false;
    while (num_unused_hits > 0 && !hcalJetClusteringFinished)
    {
    
        // Experimental vars to track energy per dR
        float hcal_seed_pt = 0.;
        float hcal_seed_eta = -999;
        float hcal_seed_phi = -999;
        int hcal_seed_iEta = -999;
        int hcal_seed_iPhi = -999;
        float hcal_seed_energy = 0;
        float hcal_dR1T = 0.;
        float hcal_dR2T = 0.;
        float hcal_dR3T = 0.;
        float hcal_dR4T = 0.;
        float hcal_dR5T = 0.;
        float hcal_2x2_1 = 0.;
        float hcal_2x2_2 = 0.;
        float hcal_2x2_3 = 0.;
        float hcal_2x2_4 = 0.;
        float hcal_nHits = 0.;

        reco::Candidate::PolarLorentzVector hcalJet( 0., 0., 0., 0.);

        // First find highest ET HCAL TP and use to seed the 11x11 HCAL Jet
        for (auto &hcalHit : hcalhits)
        {

            if (hcalHit.stale) continue; // skip hcalHits which are already used

            if (hcalJet.pt() == 0.0) // this is the first hcalHit to seed the jet
            {
                // Check if the leading unused HCAL hit has pT < min for seeding a jet.
                // If so, stop HCAL jet clustering
                if (hcalHit.pt() < EtMinForSeedHit)
                {
                    hcalJetClusteringFinished = true;
                    continue;
                }
                num_unused_hits--;
                hcal_nHits++;
                hcalJet.SetPt( hcalHit.pt() );
                hcalJet.SetEta( hcalHit.position.eta() );
                hcalJet.SetPhi( hcalHit.position.phi() );
                hcalJet.SetM( 0. );
                hcalHit.stale = true;
                hcal_seed_energy = hcalHit.energy;
                hcal_seed_pt = hcalHit.pt();
                hcal_seed_eta = hcalHit.position.eta();
                hcal_seed_phi = hcalHit.position.phi();
                hcal_seed_iEta = hcalHit.hcal_iEta;
                hcal_seed_iPhi = hcalHit.hcal_iPhi;
                //std::cout << " ----- input p4 " << l1eg.pt() << " : " << l1eg.eta() << " : " << l1eg.phi() << " : " << l1eg.M() << std::endl;
                //std::cout << " ----- jet seed p4 " << hcalJet.pt() << " : " << hcalJet.eta() << " : " << hcalJet.phi() << " : " << hcalJet.M()<< std::endl;

                // Need to add the seed energy to the dR rings
                hcal_dR1T += hcalHit.energy;
                hcal_dR2T += hcalHit.energy;
                hcal_dR3T += hcalHit.energy;
                hcal_dR4T += hcalHit.energy;
                hcal_dR5T += hcalHit.energy;

                // Some discrimination vars, 2x2s including central seed
                hcal_2x2_1 += hcalHit.energy;
                hcal_2x2_2 += hcalHit.energy;
                hcal_2x2_3 += hcalHit.energy;
                hcal_2x2_4 += hcalHit.energy;
                continue;
            }

            // Unused hcalHits which are not the initial seed
            int hit_iPhi = hcalHit.hcal_iPhi;
            int d_iEta = hcal_seed_iEta - hcalHit.hcal_iEta;
            int d_iPhi = hcalTower_diPhi( hcal_seed_iPhi, hit_iPhi );
            if ( abs( d_iEta ) <= 5 && abs( d_iPhi ) <= 5 ) // 11x11 HCAL Trigger Towers
            {
                //std::cout << "\nSeed eta, new hit eta: " << hcal_seed_eta << ":" << hcalHit.position.eta() << std::endl;
                //std::cout << " - Seed iEta, new hit iEta: " << hcal_seed_iEta << ":" << hcalHit.hcal_iEta << std::endl;
                //std::cout << " - Seed phi, new hit phi: " << hcal_seed_phi << ":" << hcalHit.position.phi() << std::endl;
                //std::cout << " - Seed iPhi, new hit iPhi: " << hcal_seed_iPhi << ":" << hcalHit.hcal_iPhi << std::endl;
                num_unused_hits--;
                hcal_nHits++;
                //std::cout << " ------- input jet p4 " << hcalJet.pt() << " : " << hcalJet.eta() << " : " << hcalJet.phi() << " : " << hcalJet.M()<< std::en     dl;
                reco::Candidate::PolarLorentzVector hcalP4( hcalHit.pt(), hcalHit.position.eta(), hcalHit.position.phi(), 0.);
                hcalJet += hcalP4;
                //std::cout << " ------- resulting jet p4 " << hcalJet.pt() << " : " << hcalJet.eta() << " : " << hcalJet.phi() << " : " << hcalJet.M()<< std     ::endl;
                hcalHit.stale = true;

                if ( abs( d_iEta ) <= 1    && abs( d_iPhi ) <= 1)   hcal_dR1T += hcalP4.energy();
                if ( abs( d_iEta ) <= 2    && abs( d_iPhi ) <= 2)   hcal_dR2T += hcalP4.energy();
                if ( abs( d_iEta ) <= 3    && abs( d_iPhi ) <= 3)   hcal_dR3T += hcalP4.energy();
                if ( abs( d_iEta ) <= 4    && abs( d_iPhi ) <= 4)   hcal_dR4T += hcalP4.energy();
                if ( abs( d_iEta ) <= 5    && abs( d_iPhi ) <= 5)   hcal_dR5T += hcalP4.energy();

                // Some discrimination vars, 2x2s including central seed
                if ( ( d_iEta == 0 || d_iEta == 1 )  &&  ( d_iPhi == 0 || d_iPhi == 1 ) )    hcal_2x2_1 += hcalP4.energy();
                if ( ( d_iEta == 0 || d_iEta == 1 )  &&  ( d_iPhi == 0 || d_iPhi == -1 ) )   hcal_2x2_2 += hcalP4.energy();
                if ( ( d_iEta == 0 || d_iEta == -1 )  &&  ( d_iPhi == 0 || d_iPhi == 1 ) )   hcal_2x2_3 += hcalP4.energy();
                if ( ( d_iEta == 0 || d_iEta == -1 )  &&  ( d_iPhi == 0 || d_iPhi == -1 ) )  hcal_2x2_4 += hcalP4.energy();
            }
        }

        // Total jet pt for sorting, then seed info second, then energy info
        std::vector< float > hcalInfo = { (float)hcalJet.pt(), hcal_seed_pt, 
                hcal_seed_eta, hcal_seed_phi, (float)hcal_seed_iEta, (float)hcal_seed_iPhi,
                hcal_seed_energy, hcal_dR1T, hcal_dR2T, hcal_dR3T, hcal_dR4T, 
                hcal_dR5T, hcal_2x2_1, hcal_2x2_2, hcal_2x2_3, hcal_2x2_4,
                hcal_nHits};

        jetClustersHcalInfo.push_back( hcalInfo );

        // store HCAL only jets
        hcalJetClusters.push_back( hcalJet );

        // And a copy for further clustering with L1EGs
        jetClusters.push_back( hcalJet );

    } // end while loop of HCAL TP clustering
        



    // Sort hcalJetClusters so we can begin with the highest pt for next step of jet clustering
    // All three must be sorted to keep same ordering
    std::sort(begin(hcalJetClusters), end(hcalJetClusters), [](const reco::Candidate::PolarLorentzVector& a,
            const reco::Candidate::PolarLorentzVector& b){return a.pt() > b.pt();});
    std::sort(begin(jetClusters), end(jetClusters), [](const reco::Candidate::PolarLorentzVector& a,
            const reco::Candidate::PolarLorentzVector& b){return a.pt() > b.pt();});
    std::sort(begin(jetClustersHcalInfo), end(jetClustersHcalInfo), [](const std::vector< float >& a,
            const std::vector< float >& b){return a.at(0) > b.at(0);});







    /**************************************************************************
 * Progress to adding L1EGs built from ECAL TPs  11x11 grid.
 * Recall, for 11x11 trigger towers gives diameter 0.957
 ******************************************************************************/


    // Make simple L1objects from the L1EG input collection with marker for 'stale'
    // FIXME could later add quality criteria here to help differentiate likely
    // photons/electrons vs. pions. This could be helpful for L1CaloTaus
    std::vector< simpleL1obj > crystalClustersVect;
    for (auto EGammaCand : crystalClusters)
    {
        simpleL1obj l1egObj;
        l1egObj.SetP4(EGammaCand.pt(), EGammaCand.eta(), EGammaCand.phi(), 0.);
        l1egObj.iEta = EGammaCand.GetExperimentalParam("seed_iEta");
        l1egObj.iPhi = EGammaCand.GetExperimentalParam("seed_iPhi");
        l1egObj.passesStandaloneWP = EGammaCand.standaloneWP();
        l1egObj.passesTrkMatchWP = EGammaCand.looseL1TkMatchWP();
        crystalClustersVect.push_back( l1egObj );
    }

    // Sorting is unnecessary as we're matching to already built HCAL Jets
    // but it is interesting to know highest pt L1EG, so sort either way
    // Sort clusters so we can always pick highest pt cluster to begin with in our jet clustering
    std::sort(begin(crystalClustersVect), end(crystalClustersVect), [](const simpleL1obj& a,
            simpleL1obj& b){return a.pt() > b.pt();});



    // Cluster together the L1EGs around existing HCAL Jet
    // Cluster within dEta/dPhi 0.5 which is very close to 0.4785 = 0.957/2
    std::cout << " - Input L1EGs: " << crystalClustersVect.size() << std::endl;
    // Keep a copy of the ecal-based 4-vector for future diagnosis
    std::vector< reco::Candidate::PolarLorentzVector > ecalJetClusters;
    int cnt = 0; // For track which jetCand we are on and mapping that to the HCAL jet and jet info vectors
    for (auto &jetCand : jetClusters)
    {

        // For tracking ecal energy spread
        float ecal_leading_pt = 0.;
        float ecal_leading_energy = 0.;
        float ecal_leading_eta = -99;
        float ecal_leading_phi = -99;
        float ecal_dR0p1_leading = 0.;
        float ecal_dR0p05 = 0.;
        float ecal_dR0p075 = 0.;
        float ecal_dR0p1 = 0.;
        float ecal_dR0p125 = 0.;
        float ecal_dR0p15 = 0.;
        float ecal_dR0p2 = 0.;
        float ecal_dR0p3 = 0.;
        float ecal_dR0p4 = 0.;
        float ecal_dR0p5 = 0.;
        float ecal_nL1EGs = 0.;
        float ecal_nL1EGs_standalone = 0.;
        float ecal_nL1EGs_trkMatch = 0.;


        reco::Candidate::PolarLorentzVector ecalJet( 0., 0., 0., 0.);

        // We are pT ordered so we will always begin with the highest pT L1EG
        simpleL1obj leadingL1EG;
        for (auto &l1eg : crystalClustersVect)
        {

            if (l1eg.stale) continue; // skip L1EGs which are already used

            // skip L1EGs outside the dEta/dPhi 0.5 range
            // cluster w.r.t. HCAL seed so the position doesn't change for every L1EG
            float d_eta = jetClustersHcalInfo[cnt].at(2) - l1eg.eta(); // See mapping of hcalInfo
            float d_phi = reco::deltaPhi( jetClustersHcalInfo[cnt].at(3), l1eg.phi() ); // See mapping of hcalInfo
            float d_eta_to_leading = -99;
            float d_phi_to_leading = -99;
            if ( fabs( d_eta ) > 0.5 || fabs( d_phi ) > 0.5 ) continue;

            if (ecalJet.pt() == 0.0) // this is the first L1EG to seed the ecal jet
            {
                ecalJet.SetPt( l1eg.pt() );
                ecalJet.SetEta( l1eg.eta() );
                ecalJet.SetPhi( l1eg.phi() );
                ecalJet.SetM( l1eg.M() );
                ecal_leading_pt = l1eg.pt();
                ecal_leading_energy = l1eg.GetP4().energy();
                ecal_leading_eta = l1eg.eta();
                ecal_leading_phi = l1eg.phi();
                leadingL1EG = l1eg;
                //std::cout << " --- initial jet seed " << cnt << std::endl;
                //std::cout << " ----- input cal jet p4 " << jetCand.pt() << " : " << jetCand.eta() << " : " << jetCand.phi() << std::endl;
                //std::cout << " ----- ecal jet seed p4 " << ecalJet.pt() << " : " << ecalJet.eta() << " : " << ecalJet.phi() <<  std::endl;
            }
            else // subsequent L1EGs
            {
                ecalJet += l1eg.GetP4();
                d_eta_to_leading = ecal_leading_eta - l1eg.eta();
                d_phi_to_leading = reco::deltaPhi( ecal_leading_phi, l1eg.GetP4().phi() );
            }

            // For all including the seed and subsequent L1EGs
            ecal_nL1EGs++;
            if (l1eg.passesStandaloneWP ) ecal_nL1EGs_standalone++;
            if (l1eg.passesTrkMatchWP ) ecal_nL1EGs_trkMatch++;
            l1eg.stale = true;

            // Unused L1EGs which are not the initial ecal jet seed
            //std::cout << " ------- input jet p4 " << ecalJet.pt() << " : " << ecalJet.eta() << " : " << ecalJet.phi() << " : " << ecalJet.M()<< std::endl;
            //std::cout << " ------- input l1eg: " << cnt << " : p4 " << l1eg.pt() << " : " << l1eg.eta() << " : " << l1eg.phi() << " : " << l1eg.M() << std::endl;
            //std::cout << " ------- resulting jet p4 " << ecalJet.pt() << " : " << ecalJet.eta() << " : " << ecalJet.phi() << " : " << ecalJet.M()<< std::endl;

            // Make energy sums in rings, 1 type is centered on highest pT L1EG
            if ( fabs( d_eta_to_leading ) < 0.1   && fabs( d_phi_to_leading ) < 0.1  )  ecal_dR0p1_leading   += l1eg.GetP4().energy();
            // Other type is centered on the HCAL jet center
            if ( fabs( d_eta ) < 0.05  && fabs( d_phi ) < 0.05 )  ecal_dR0p05  += l1eg.GetP4().energy();
            if ( fabs( d_eta ) < 0.075 && fabs( d_phi ) < 0.075)  ecal_dR0p075 += l1eg.GetP4().energy();
            if ( fabs( d_eta ) < 0.1   && fabs( d_phi ) < 0.1  )  ecal_dR0p1   += l1eg.GetP4().energy();
            if ( fabs( d_eta ) < 0.125 && fabs( d_phi ) < 0.125)  ecal_dR0p125 += l1eg.GetP4().energy();
            if ( fabs( d_eta ) < 0.15  && fabs( d_phi ) < 0.15 )  ecal_dR0p15  += l1eg.GetP4().energy();
            if ( fabs( d_eta ) < 0.2   && fabs( d_phi ) < 0.2  )  ecal_dR0p2   += l1eg.GetP4().energy();
            if ( fabs( d_eta ) < 0.3   && fabs( d_phi ) < 0.3  )  ecal_dR0p3   += l1eg.GetP4().energy();
            if ( fabs( d_eta ) < 0.4   && fabs( d_phi ) < 0.4  )  ecal_dR0p4   += l1eg.GetP4().energy();
            if ( fabs( d_eta ) < 0.5   && fabs( d_phi ) < 0.5  )  ecal_dR0p5   += l1eg.GetP4().energy();
        }

        // For total p4()
        jetCand += ecalJet;

        params["hcal_pt"] = hcalJetClusters[cnt].pt();
        params["hcal_eta"] = hcalJetClusters[cnt].eta();
        params["hcal_phi"] = hcalJetClusters[cnt].phi();
        params["hcal_mass"] = hcalJetClusters[cnt].mass();
        params["hcal_energy"] = hcalJetClusters[cnt].energy();

        params["hcal_jet_pt"] = jetClustersHcalInfo[cnt].at(0);
        params["hcal_seed_pt"] = jetClustersHcalInfo[cnt].at(1);
        params["hcal_seed_eta"] = jetClustersHcalInfo[cnt].at(2);
        params["hcal_seed_phi"] = jetClustersHcalInfo[cnt].at(3);
        params["hcal_seed_iEta"] = jetClustersHcalInfo[cnt].at(4);
        params["hcal_seed_iPhi"] = jetClustersHcalInfo[cnt].at(5);
        params["hcal_seed_energy"] = jetClustersHcalInfo[cnt].at(6);
        params["hcal_dR1T"] = jetClustersHcalInfo[cnt].at(7);
        params["hcal_dR2T"] = jetClustersHcalInfo[cnt].at(8);
        params["hcal_dR3T"] = jetClustersHcalInfo[cnt].at(9);
        params["hcal_dR4T"] = jetClustersHcalInfo[cnt].at(10);
        params["hcal_dR5T"] = jetClustersHcalInfo[cnt].at(11);
        params["hcal_2x2_1"] = jetClustersHcalInfo[cnt].at(12);
        params["hcal_2x2_2"] = jetClustersHcalInfo[cnt].at(13);
        params["hcal_2x2_3"] = jetClustersHcalInfo[cnt].at(14);
        params["hcal_2x2_4"] = jetClustersHcalInfo[cnt].at(15);
        params["hcal_nHits"] = jetClustersHcalInfo[cnt].at(16);

        // See mapping of hcalInfo
        reco::Candidate::PolarLorentzVector hcalSeed(
            jetClustersHcalInfo[cnt].at(1),
            jetClustersHcalInfo[cnt].at(2),
            jetClustersHcalInfo[cnt].at(3), 0.);

        // return -9 for energy and dR values for ecalJet as defaults
        float hovere = -9;
        params["deltaR_ecal_vs_jet"] = -9;
        params["deltaR_ecal_vs_hcal"] = -9; 
        params["deltaR_ecal_vs_hcal_seed"] = -9;
        params["deltaR_ecal_lead_vs_ecal"] = -9;
        params["deltaR_ecal_lead_vs_jet"] = -9;
        if (ecalJet.pt() > 0.0)
        {
            params["deltaR_ecal_vs_jet"] = reco::deltaR( ecalJet, jetCand );
            params["deltaR_ecal_vs_hcal"] = reco::deltaR( ecalJet, hcalJetClusters[cnt] );
            params["deltaR_ecal_vs_hcal_seed"] = reco::deltaR( ecalJet, hcalSeed );
            params["deltaR_ecal_lead_vs_ecal"] = reco::deltaR( ecalJet, leadingL1EG );
            params["deltaR_ecal_lead_vs_jet"] = reco::deltaR( jetCand, leadingL1EG );
            hovere = hcalJetClusters[cnt].energy() / ecalJet.energy();
        }
        params["deltaR_hcal_vs_jet"] = reco::deltaR( hcalJetClusters[cnt], jetCand );
        params["deltaR_hcal_vs_hcal_seed"] = reco::deltaR( hcalJetClusters[cnt], hcalSeed );
        cnt++;


        params["ecal_leading_pt"] =     ecal_leading_pt;
        params["ecal_leading_eta"] =    ecal_leading_eta;
        params["ecal_leading_phi"] =    ecal_leading_phi;
        params["ecal_leading_energy"] = ecal_leading_energy;
        params["ecal_dR0p1_leading"] =  ecal_dR0p1_leading;
        params["ecal_dR0p05"] =         ecal_dR0p05;
        params["ecal_dR0p075"] =        ecal_dR0p075;
        params["ecal_dR0p1"] =          ecal_dR0p1;
        params["ecal_dR0p125"] =        ecal_dR0p125;
        params["ecal_dR0p15"] =         ecal_dR0p15;
        params["ecal_dR0p2"] =          ecal_dR0p2;
        params["ecal_dR0p3"] =          ecal_dR0p3;
        params["ecal_dR0p4"] =          ecal_dR0p4;
        params["ecal_dR0p5"] =          ecal_dR0p5;
        params["ecal_nL1EGs"] =         ecal_nL1EGs;
        params["ecal_nL1EGs_standalone"] =  ecal_nL1EGs_standalone;
        params["ecal_nL1EGs_trkMatch"] =    ecal_nL1EGs_trkMatch;

        params["ecal_pt"] = ecalJet.pt();
        params["ecal_eta"] = ecalJet.eta();
        params["ecal_phi"] = ecalJet.phi();
        params["ecal_mass"] = ecalJet.mass();
        params["ecal_energy"] = ecalJet.energy();

        params["jet_pt"] = jetCand.pt();
        params["jet_eta"] = jetCand.eta();
        params["jet_phi"] = jetCand.phi();
        params["jet_mass"] = jetCand.mass();
        params["jet_energy"] = jetCand.energy();

        float calibratedPt = -1;
        float ECalIsolation = -1; // Need to loop over 5x5 crystals of unclustered energy
        float totalPtPUcorr = -1;
        l1slhc::L1CaloJet caloJet(jetCand, calibratedPt, hovere, ECalIsolation, totalPtPUcorr);
        caloJet.SetExperimentalParams(params);

        L1CaloTausNoCuts->push_back( caloJet );
        // Same for the moment...
        L1CaloTausWithCuts->push_back( caloJet );


    } // end jetClusters loop
    std::cout << " - resulting # jets: " << jetClusters.size() << std::endl;







    // Per subdetector breakdown
    //for (size_t i = 0; i < jetClusters.size(); i++ )
    //{
    //    std::cout << " --- Jet " << i <<  std::endl;
    //    std::cout << " ----- ECAL      " << ecalJetClusters[i].pt() << " : " << ecalJetClusters[i].eta() << " : " << ecalJetClusters[i].phi() << " : " << ecalJetClusters[i].M()<< std::endl;
    //    std::cout << " ----- HCAL      " << hcalJetClusters[i].pt() << " : " << hcalJetClusters[i].eta() << " : " << hcalJetClusters[i].phi() << " : " << hcalJetClusters[i].M()<< std::endl;
    //    std::cout << " ----- Jet Cand  " << jetClusters[i].pt() << " : " << jetClusters[i].eta() << " : " << jetClusters[i].phi() << " : " << jetClusters[i].M()<< std::endl;
    //}


    iEvent.put(std::move(L1CaloTausNoCuts),"L1CaloTausNoCuts");
    iEvent.put(std::move(L1CaloTausWithCuts), "L1CaloTausWithCuts" );
    //iEvent.put(std::move(L1CaloClusterCollectionWithCuts), "L1CaloClusterCollectionWithCuts" );
    iEvent.put(std::move(L1CaloClusterCollectionBXVWithCuts),"L1CaloClusterCollectionBXVWithCuts");

















    
    // Clustering algorithm
    //while(true)
    //{
    //    // Find highest pt hit (that's not already used)
    //    SimpleCaloHit centerhit;
    //    for(const auto& hit : ecalhits)
    //    {
    //        if ( !hit.stale && hit.pt() > centerhit.pt() )
    //        {
    //            centerhit = hit;
    //        }
    //    }
    //    // If we are less than 1GeV (configurable with EtMinForSeedHit) 
    //    // or out of hits (i.e. when centerhit is default constructed) we stop
    //    if ( centerhit.pt() < EtMinForSeedHit ) break;
    //    if ( debug ) std::cout << "-------------------------------------" << std::endl;
    //    if ( debug ) std::cout << "New cluster: center crystal pt = " << centerhit.pt() << std::endl;

    //    // Experimental parameters, don't want to bother with hardcoding them in data format
    //    std::map<std::string, float> params;
    //    
    //    // Find the energy-weighted average position,
    //    // calculate isolation parameter,
    //    // calculate pileup-corrected pt,
    //    // and quantify likelihood of a brem
    //    GlobalVector weightedPosition;
    //    GlobalVector ECalPileUpVector;
    //    float totalEnergy = 0.;
    //    float ECalIsolation = 0.;
    //    float ECalPileUpEnergy = 0.;
    //    float e2x2_1 = 0.;
    //    float e2x2_2 = 0.;
    //    float e2x2_3 = 0.;
    //    float e2x2_4 = 0.;
    //    float e2x2 = 0.;
    //    float e2x5_1 = 0.;
    //    float e2x5_2 = 0.;
    //    float e2x5 = 0.;
    //    float e5x5 = 0.;
    //    float e3x5 = 0.;
    //    bool standaloneWP;
    //    bool electronWP98;
    //    bool looseL1TkMatchWP;
    //    bool photonWP80;
    //    bool electronWP90;
    //    bool passesStage2Eff;
    //    std::vector<float> crystalPt;
    //    std::map<int, float> phiStrip;
    //    //std::cout << " -- iPhi: " << ehit.id.iphi() << std::endl;
    //    //std::cout << " -- iEta: " << ehit.id.ieta() << std::endl;

    //    for(auto& hit : ecalhits)
    //    {

    //        if ( !hit.stale &&
    //                ( (!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 2 && abs(hit.diphi(centerhit)) < 3)
    //                 || (centerhit.isEndcapHit && hit.distanceTo(centerhit) < 3.5*1.41 ) )) // endcap crystals are 30mm on a side, 3.5*sqrt(2) cm radius should enclose 3x3
    //        {
    //            weightedPosition += hit.position*hit.energy;
    //            totalEnergy += hit.energy;
    //            hit.stale = true;
    //            crystalPt.push_back(hit.pt());
    //            if ( debug && hit == centerhit )
    //                std::cout << "\x1B[32m"; // green hilight
    //            if ( debug && hit.isEndcapHit ) std::cout <<
    //                "\tCrystal pt=" << hit.pt() <<
    //                ", eta=" << hit.position.eta() <<
    //                ", phi=" << hit.position.phi() << "\x1B[0m" << std::endl;
    //            else if ( debug ) std::cout <<
    //                "\tCrystal (" << hit.dieta(centerhit) << "," << hit.diphi(centerhit) <<
    //                ") pt=" << hit.pt() <<
    //                ", eta=" << hit.position.eta() <<
    //                ", phi=" << hit.position.phi() << "\x1B[0m" << std::endl;
    //        }

    //        if ( abs(hit.dieta(centerhit)) == 0 && abs(hit.diphi(centerhit)) <= 7 )
    //        {
    //            phiStrip[hit.diphi(centerhit)] = hit.pt();
    //        }

    //        // Build 5x5
    //        if ( abs(hit.dieta(centerhit)) < 3 && abs(hit.diphi(centerhit)) < 3 )
    //        {
    //            e5x5 += hit.energy;
    //        }

    //        // Build 3x5
    //        if ( abs(hit.dieta(centerhit)) < 2 && abs(hit.diphi(centerhit)) < 3 )
    //        {
    //            e3x5 += hit.energy;
    //        }

    //        // Build E2x5
    //        if ( (hit.dieta(centerhit) == 0 || hit.dieta(centerhit) == 1) && abs(hit.diphi(centerhit)) < 3 )
    //        {
    //            e2x5_1 += hit.energy;
    //        }
    //        if ( (hit.dieta(centerhit) == 0 || hit.dieta(centerhit) == -1) && abs(hit.diphi(centerhit)) < 3 )
    //        {
    //            e2x5_2 += hit.energy;
    //        }
    //        e2x5 = TMath::Max( e2x5_1, e2x5_2 );

    //        // Build 2x2, highest energy 2x2 square containing the seed
    //        if ( (hit.dieta(centerhit) == 0 || hit.dieta(centerhit) == 1) && (hit.diphi(centerhit) == 0 || hit.diphi(centerhit) == 1) )
    //        {
    //            e2x2_1 += hit.energy;
    //        }
    //        if ( (hit.dieta(centerhit) == 0 || hit.dieta(centerhit) == 1) && (hit.diphi(centerhit) == 0 || hit.diphi(centerhit) == -1) )
    //        {
    //            e2x2_2 += hit.energy;
    //        }
    //        if ( (hit.dieta(centerhit) == 0 || hit.dieta(centerhit) == -1) && (hit.diphi(centerhit) == 0 || hit.diphi(centerhit) == 1) )
    //        {
    //            e2x2_3 += hit.energy;
    //        }
    //        if ( (hit.dieta(centerhit) == 0 || hit.dieta(centerhit) == -1) && (hit.diphi(centerhit) == 0 || hit.diphi(centerhit) == -1) )
    //        {
    //            e2x2_4 += hit.energy;
    //        }

    //        e2x2 = TMath::Max( e2x2_1, e2x2_2 );
    //        e2x2 = TMath::Max( e2x2, e2x2_3 );
    //        e2x2 = TMath::Max( e2x2, e2x2_4 );
    //        params["E2x2"] = e2x2;
    //        params["E2x5"] = e2x5;
    //        params["E3x5"] = e3x5;
    //        params["E5x5"] = e5x5;

    //        // Isolation and pileup must not use hits used in the cluster
    //        // As for the endcap hits, well, as far as this algorithm is concerned, caveat emptor...
    //        if ( !(!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 2 && abs(hit.diphi(centerhit)) < 3)
    //              && !(centerhit.isEndcapHit && hit.distanceTo(centerhit) < 3.5*1.41 ) )
    //        {
    //            if ( (!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 14 && abs(hit.diphi(centerhit)) < 14)
    //                  || (centerhit.isEndcapHit && hit.distanceTo(centerhit) < 42. ))
    //            {
    //                ECalIsolation += hit.pt();
    //                if ( hit.pt() > 1. )
    //                    params["nIsoCrystals1"]++;
    //            }
    //            if ( hit.pt() < 5. &&
    //                  ( (!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 7 && abs(hit.diphi(centerhit)) < 57 )
    //                    || (centerhit.isEndcapHit && hit.distanceTo(centerhit) < 50.) ))
    //            {
    //                ECalPileUpEnergy += hit.energy;
    //                ECalPileUpVector += hit.position;
    //            }
    //        }
    //    }
    //    params["uncorrectedE"] = totalEnergy;
    //    params["uncorrectedPt"] = totalEnergy*sin(weightedPosition.theta());

    //    // phi strip params
    //    // lambda returns size of contiguous strip, one-hole strip
    //    auto countStrip = [&phiStrip](float threshold) -> std::pair<float, float>
    //    {
    //        int nContiguous = 1;
    //        int nOneHole = 1;
    //        bool firstHole = false;
    //        for(int i=1; i<=7; ++i)
    //        {
    //            if ( phiStrip[i] > threshold && !firstHole )
    //            {
    //                nContiguous++;
    //                nOneHole++;
    //            }
    //            else if ( phiStrip[i] > threshold )
    //                nOneHole++;
    //            else if ( !firstHole )
    //                firstHole = true;
    //            else
    //                break;
    //        }
    //        firstHole = false;
    //        for(int i=-1; i>=-7; --i)
    //        {
    //            if ( phiStrip[i] > threshold && !firstHole )
    //            {
    //                nContiguous++;
    //                nOneHole++;
    //            }
    //            else if ( phiStrip[i] > threshold )
    //                nOneHole++;
    //            else if ( !firstHole )
    //                firstHole = true;
    //            else
    //                break;
    //        }
    //        return std::make_pair<float, float>(nContiguous, nOneHole);
    //    };
    //    auto zeropair = countStrip(0.);
    //    params["phiStripContiguous0"] = zeropair.first;
    //    params["phiStripOneHole0"] = zeropair.second;
    //    auto threepair = countStrip(0.03*totalEnergy);
    //    params["phiStripContiguous3p"] = threepair.first;
    //    params["phiStripOneHole3p"] = threepair.second;


    //    // no need to rescale weightedPosition if we only use theta
    //    float correctedTotalPt = totalEnergy*sin(weightedPosition.theta());
    //    params["avgIsoCrystalE"] = (params["nIsoCrystals1"] > 0.) ? ECalIsolation/params["nIsoCrystals1"] : 0.;
    //    ECalIsolation /= params["uncorrectedPt"];
    //    float totalPtPUcorr = params["uncorrectedPt"] - ECalPileUpEnergy*sin(ECalPileUpVector.theta())/19.;
    //    float bremStrength = params["uncorrectedPt"] / correctedTotalPt;

    //    if ( debug ) std::cout << "Weighted position eta = " << weightedPosition.eta() << ", phi = " << weightedPosition.phi() << std::endl;
    //    if ( debug ) std::cout << "Uncorrected Total energy = " << params["uncorrectedE"] << ", total pt = " << params["uncorrectedPt"] << std::endl;
    //    if ( debug ) std::cout << "Total energy = " << totalEnergy << ", total pt = " << correctedTotalPt << std::endl;
    //    if ( debug ) std::cout << "Isolation: " << ECalIsolation << std::endl;


    //    // Calibrate L1EG pT to match Stage-2 (Phase-I) calibrations
    //    // NOTE: working points are defined with respect to normal correctedTotalPt
    //    // not to calibrated pT
    //    float calibratedPt; 
    //    calibratedPt = correctedTotalPt * ( ptAdjustFunc.Eval( correctedTotalPt ) );


    //    // Calculate H/E if we have hcalhits
    //    // else fill with -1. so no one gets confused
    //    // and thinks 0. is H/E
    //    float hcalEnergy = 0.;
    //    float hovere;
    //    float hovereCalibPt;
    //    float sineTerm = sin(weightedPosition.theta());
    //    float minimum = 1e-5;
    //    float calibratedE = calibratedPt/ TMath::Max( sineTerm, minimum);

    //    float hcal_dR0p05 = 0.;
    //    float hcal_dR0p075 = 0.;
    //    float hcal_dR0p1 = 0.;
    //    float hcal_dR0p125 = 0.;
    //    float hcal_dR0p15 = 0.;
    //    float hcal_dR0p2 = 0.;
    //    float hcal_dR0p3 = 0.;
    //    float hcal_dR0p4 = 0.;
    //    float hcal_dR0p5 = 0.;

    //    float hcal_dR0p05_nTowers = 0.;
    //    float hcal_dR0p075_nTowers = 0.;
    //    float hcal_dR0p1_nTowers = 0.;
    //    float hcal_dR0p125_nTowers = 0.;
    //    float hcal_dR0p15_nTowers = 0.;
    //    float hcal_dR0p2_nTowers = 0.;
    //    float hcal_dR0p3_nTowers = 0.;
    //    float hcal_dR0p4_nTowers = 0.;
    //    float hcal_dR0p5_nTowers = 0.;

    //    if (hcalhits.size() > 0) {
    //      int cnt = 0;
    //      for(const auto& hit : hcalhits)
    //      {
    //          cnt++;
    //          //std::cout << " - " << cnt << "  dEta: " << fabs(hit.deta(centerhit)) << "    dPhi: " << fabs(hit.dphi(centerhit)) << std::endl;
    //          if ( fabs(hit.deta(centerhit)) < 0.15 && fabs(hit.dphi(centerhit)) < 0.15 )
    //          {
    //              hcalEnergy += hit.energy;
    //              //std::cout << " --- " << cnt << "  hit energy: " << hit.energy << "    total HCAL: " << hcalEnergy << std::endl;
    //          }
    //          if ( fabs(hit.deta(centerhit)) < 0.05 && fabs(hit.dphi(centerhit)) < 0.05 )
    //          {
    //              hcal_dR0p05 += hit.energy;
    //              hcal_dR0p05_nTowers += 1;
    //          }
    //          if ( fabs(hit.deta(centerhit)) < 0.075 && fabs(hit.dphi(centerhit)) < 0.075 )
    //          {
    //              hcal_dR0p075 += hit.energy;
    //              hcal_dR0p075_nTowers += 1;
    //          }
    //          if ( fabs(hit.deta(centerhit)) < 0.10 && fabs(hit.dphi(centerhit)) < 0.10 )
    //          {
    //              hcal_dR0p1 += hit.energy;
    //              hcal_dR0p1_nTowers += 1;
    //          }
    //          if ( fabs(hit.deta(centerhit)) < 0.125 && fabs(hit.dphi(centerhit)) < 0.125 )
    //          {
    //              hcal_dR0p125 += hit.energy;
    //              hcal_dR0p125_nTowers += 1;
    //          }
    //          if ( fabs(hit.deta(centerhit)) < 0.15 && fabs(hit.dphi(centerhit)) < 0.15 )
    //          {
    //              hcal_dR0p15 += hit.energy;
    //              hcal_dR0p15_nTowers += 1;
    //          }
    //          if ( fabs(hit.deta(centerhit)) < 0.20 && fabs(hit.dphi(centerhit)) < 0.20 )
    //          {
    //              hcal_dR0p2 += hit.energy;
    //              hcal_dR0p2_nTowers += 1;
    //          }
    //          if ( fabs(hit.deta(centerhit)) < 0.30 && fabs(hit.dphi(centerhit)) < 0.30 )
    //          {
    //              hcal_dR0p3 += hit.energy;
    //              hcal_dR0p3_nTowers += 1;
    //          }
    //          if ( fabs(hit.deta(centerhit)) < 0.40 && fabs(hit.dphi(centerhit)) < 0.40 )
    //          {
    //              hcal_dR0p4 += hit.energy;
    //              hcal_dR0p4_nTowers += 1;
    //          }
    //          if ( fabs(hit.deta(centerhit)) < 0.50 && fabs(hit.dphi(centerhit)) < 0.50 )
    //          {
    //              hcal_dR0p5 += hit.energy;
    //              hcal_dR0p5_nTowers += 1;
    //          }
    //      }
    //      params["hcal_dR0p05"] = hcal_dR0p05*sineTerm;
    //      params["hcal_dR0p075"] = hcal_dR0p075*sineTerm;
    //      params["hcal_dR0p1"] = hcal_dR0p1*sineTerm;
    //      params["hcal_dR0p125"] = hcal_dR0p125*sineTerm;
    //      params["hcal_dR0p15"] = hcal_dR0p15*sineTerm;
    //      params["hcal_dR0p2"] = hcal_dR0p2*sineTerm;
    //      params["hcal_dR0p3"] = hcal_dR0p3*sineTerm;
    //      params["hcal_dR0p4"] = hcal_dR0p4*sineTerm;
    //      params["hcal_dR0p5"] = hcal_dR0p5*sineTerm;
    //      params["hcal_dR0p05_nTowers"] = hcal_dR0p05_nTowers;
    //      params["hcal_dR0p075_nTowers"] = hcal_dR0p075_nTowers;
    //      params["hcal_dR0p1_nTowers"] = hcal_dR0p1_nTowers;
    //      params["hcal_dR0p125_nTowers"] = hcal_dR0p125_nTowers;
    //      params["hcal_dR0p15_nTowers"] = hcal_dR0p15_nTowers;
    //      params["hcal_dR0p2_nTowers"] = hcal_dR0p2_nTowers;
    //      params["hcal_dR0p3_nTowers"] = hcal_dR0p3_nTowers;
    //      params["hcal_dR0p4_nTowers"] = hcal_dR0p4_nTowers;
    //      params["hcal_dR0p5_nTowers"] = hcal_dR0p5_nTowers;
    //      hovere = hcalEnergy/params["uncorrectedE"];
    //      hovereCalibPt = hcalEnergy/calibratedE;
    //    }
    //    else
    //    {
    //      hovere = -1.0;
    //      hovereCalibPt = -1.0;
    //    }

    //    if ( debug && calibratedPt > 10 ) std::cout << "E: " << params["uncorrectedE"] << " CalibE: " << calibratedE << " H/E: " << hovere << "    H/E Calib: " << hovereCalibPt << std::endl;

    //    // Check if cluster passes electron or photon WPs
    //    // Note: WPs cuts are defined with respect to non-calibrated pT and non-calibrated H/E
    //    float cluster_eta = weightedPosition.eta();
    //    //standaloneWP = cluster_passes_base_cuts( correctedTotalPt, cluster_eta, ECalIsolation, e2x5, e5x5);

    //    
    //    // Form a l1slhc::L1EGCrystalCluster
    //    //reco::Candidate::PolarLorentzVector p4(correctedTotalPt, weightedPosition.eta(), weightedPosition.phi(), 0.);
    //    reco::Candidate::PolarLorentzVector p4calibrated(calibratedPt, weightedPosition.eta(), weightedPosition.phi(), 0.);
    //    l1slhc::L1EGCrystalCluster cluster(p4calibrated, calibratedPt, hovereCalibPt, ECalIsolation, centerhit.id, totalPtPUcorr, bremStrength,
    //            e2x2, e2x5, e3x5, e5x5, standaloneWP, electronWP98, photonWP80, electronWP90, looseL1TkMatchWP, passesStage2Eff);
    //    // Save pt array
    //    cluster.SetCrystalPtInfo(crystalPt);
    //    params["crystalCount"] = crystalPt.size();
    //    params["preCalibratedPt"] = correctedTotalPt;
    //    cluster.SetExperimentalParams(params);
    //    L1CaloTausNoCuts->push_back(cluster);


    //    // Save clusters passing ANY of the defined WPs
    //    if ( standaloneWP || electronWP98 || looseL1TkMatchWP || photonWP80 || electronWP90 || passesStage2Eff )
    //    {
    //        // Optional min. Et cut
    //        if ( cluster.pt() >= EtminForStore ) {
    //            L1CaloTausWithCuts->push_back(cluster);
    //            L1CaloClusterCollectionWithCuts->push_back(l1extra::L1EmParticle(p4calibrated, edm::Ref<L1GctEmCandCollection>(), 0));

    //            // BXVector l1t::Tau quality defined with respect to these WPs
    //            int quality = (standaloneWP*2^0) + (electronWP98*2^1) + (looseL1TkMatchWP*2^2) + (photonWP80*2^3) + (electronWP90*2^4) + (passesStage2Eff*2^5);
    //            L1CaloClusterCollectionBXVWithCuts->push_back(0,l1t::Tau(p4calibrated, calibratedPt, weightedPosition.eta(), weightedPosition.phi(),quality,1 ));
    //            if (debug) std::cout << "Quality: "<<  std::bitset<10>(quality) << std::endl;
    //        }
    //    }
    //}
}




//bool
//L1CaloTauProducer::cluster_passes_base_cuts(float &cluster_pt, float &cluster_eta, float &cluster_iso, float &e2x5, float &e5x5) const {
//    //return true;
//    
//    // Currently this producer is optimized based on cluster isolation and shower shape
//    // The following cut is based off of what was shown in the Phase-2 meeting
//    // 23 May 2017 from CMSSW 92X
//    // Optimization based on min ECAL TP ET = 500 MeV for inclusion
//    // Only the barrel is considered
//    if ( fabs(cluster_eta) < 1.479 )
//    {
//      
//        if ( !( 0.94 + 0.052 * TMath::Exp( -0.044 * cluster_pt ) < (e2x5 / e5x5)) )
//            return false;
//        if ( cluster_pt < 80 ) {
//            if ( !(( 0.85 + -0.0080 * cluster_pt ) > cluster_iso ) ) return false;
//        }
//        if ( cluster_pt >= 80 ) { // do flat line extension of isolation cut
//            if ( cluster_iso > 0.21 ) return false;
//        }
//        return true; // cluster passes all cuts
//    }
//    return false; // out of eta range
//}


int
L1CaloTauProducer::ecalXtal_diPhi( int &iPhi_1, int &iPhi_2 ) const
{
    // We shouldn't compare integer indices in endcap, the map is not linear
    // Logic from EBDetId::distancePhi() without the abs()
    int PI = 180;
    int result = iPhi_1 - iPhi_2;
    while (result > PI) result -= 2*PI;
    while (result <= -PI) result += 2*PI;
    return result;
}


int
L1CaloTauProducer::hcalTower_diPhi( int &iPhi_1, int &iPhi_2 ) const
{
    // 360 Crystals in full, 72 towers, half way is 36
    int PI = 36;
    int result = iPhi_1 - iPhi_2;
    while (result > PI) result -= 2*PI;
    while (result <= -PI) result += 2*PI;
    return result;
}


DEFINE_FWK_MODULE(L1CaloTauProducer);
