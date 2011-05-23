#include "RecoEcal/EgammaClusterAlgos/interface/HybridClusterAlgo.h"
#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include <iostream>
#include <map>
#include <vector>
#include <set>
#include "RecoEcal/EgammaCoreTools/interface/ClusterEtLess.h"

//The real constructor
HybridClusterAlgo::HybridClusterAlgo(double eb_str, 
				     int step, 
				     double ethres,
				     double eseed,
				     double ewing,
				     std::vector<int> v_chstatus,
				     const PositionCalc& posCalculator,
				     DebugLevel debugLevel,
				     bool dynamicEThres,
				     double eThresA,
				     double eThresB,
				     std::vector<int> severityToExclude,
				     double severityRecHitThreshold,
				     int severitySpikeId,
				     double severitySpikeThreshold,
				     bool excludeFromCluster
				     ) :
  
  eb_st(eb_str), phiSteps_(step), 
  eThres_(ethres), eThresA_(eThresA), eThresB_(eThresB),
  Eseed(eseed),  Ewing(ewing), 
  dynamicEThres_(dynamicEThres), debugLevel_(debugLevel),
  v_chstatus_(v_chstatus), v_severitylevel_(severityToExclude),severityRecHitThreshold_(severityRecHitThreshold), severitySpikeThreshold_(severitySpikeThreshold), excludeFromCluster_(excludeFromCluster)
  
  
{
  spId_ = EcalSeverityLevelAlgo::SpikeId(severitySpikeId);
  dynamicPhiRoad_ = false;
  if ( debugLevel_ == pDEBUG ) {
    //std::cout << "dynamicEThres: " << dynamicEThres_ 
    //          << " : A,B " << eThresA_ << ", " << eThresB_ << std::endl;
  }
  
  //if (dynamicPhiRoad_) phiRoadAlgo_ = new BremRecoveryPhiRoadAlgo(bremRecoveryPset);
  posCalculator_ = posCalculator;
  topo_ = new EcalBarrelHardcodedTopology();
  
  std::sort( v_chstatus_.begin(), v_chstatus_.end() );
  
  
}

// Return a vector of clusters from a collection of EcalRecHits:
void HybridClusterAlgo::makeClusters(const EcalRecHitCollection*recColl, 
				     const CaloSubdetectorGeometry*geometry,
				     reco::BasicClusterCollection &basicClusters,
				     bool regional,
				     const std::vector<EcalEtaPhiRegion>& regions,
				     const EcalChannelStatus*chStatus
				     )
{
  //clear vector of seeds
  seeds.clear();
  //clear flagged crystals
  excludedCrys_.clear();
  //clear map of supercluster/basiccluster association
  clustered_.clear();
  //clear set of used detids
  useddetids.clear();
  //clear vector of seed clusters
  seedClus_.clear();
  //Pass in a pointer to the collection.
  recHits_ = recColl;
  
  //
  //  SCShape_ = new SuperClusterShapeAlgo(recHits_, geometry);
  
  if ( debugLevel_ == pDEBUG ) {
    std::cout << "Cleared vectors, starting clusterization..." << std::endl;
    std::cout << "Purple monkey aardvark." << std::endl;
  }
  
  int nregions=0;
  if(regional) nregions=regions.size();
  
  if(!regional || nregions) {
    
    EcalRecHitCollection::const_iterator it;
    
    for (it = recHits_->begin(); it != recHits_->end(); it++){
      
      //Make the vector of seeds that we're going to use.
      //One of the few places position is used, needed for ET calculation.    
      const CaloCellGeometry *this_cell = (*geometry).getGeometry(it->id());
      GlobalPoint position = this_cell->getPosition();
      
      
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

	//Must pass seed threshold
	float ET = it->energy() * sin(position.theta());

	if (ET > eb_st) {
	  if (debugLevel_==pDEBUG) std::cout << "Seed crystal: " << std::endl;
	  // avoid seeding for anomalous channels (recoFlag based)
	  uint32_t rhFlag = (*it).recoFlag();
	  if (debugLevel_==pDEBUG) std::cout << "rhFlag: " << rhFlag << std::endl;
	  std::vector<int>::const_iterator vit = std::find( v_chstatus_.begin(), v_chstatus_.end(), rhFlag );
	  if ( vit != v_chstatus_.end() ){
	    if (excludeFromCluster_)
	      excludedCrys_.insert(it->id());
	    continue; // the recHit has to be excluded from seeding
	  }
	  int severityFlag =  EcalSeverityLevelAlgo::severityLevel( it->id(), 
								  (*recHits_), 
								  (*chStatus),
								  severityRecHitThreshold_,
								  spId_,
								  severitySpikeThreshold_);
	  std::vector<int>::const_iterator sit = std::find( v_severitylevel_.begin(), v_severitylevel_.end(), severityFlag);
	  if (debugLevel_ == pDEBUG){
	    std::cout << "found flag: " << severityFlag << std::endl;
	  }
	  
	  
	  if (sit!=v_severitylevel_.end()){ 
	    if (excludeFromCluster_)
	      excludedCrys_.insert(it->id());
	    continue;
	  }
	  seeds.push_back(*it);
	  if ( debugLevel_ == pDEBUG ){
	    std::cout << "Seed ET: " << ET << std::endl;
	    std::cout << "Seed E: " << it->energy() << std::endl;
	  }
	}
      }
    }
    
  }
  
  
  //Yay sorting.
  if ( debugLevel_ == pDEBUG )
    std::cout << "Built vector of seeds, about to sort them...";
  
  //Needs three argument sort with seed comparison operator
  sort(seeds.begin(), seeds.end(), less_mag());
  
  if ( debugLevel_ == pDEBUG )
    std::cout << "done" << std::endl;
  
  //Now to do the work.
  if ( debugLevel_ ==pDEBUG ) 
    std::cout << "About to call mainSearch...";
  mainSearch(recColl,geometry);
  if ( debugLevel_ == pDEBUG ) 
    std::cout << "done" << std::endl;
  
  //Hand the basicclusters back to the producer.  It has to 
  //put them in the event.  Then we can make superclusters.
  std::map<int, reco::BasicClusterCollection>::iterator bic; 
  for (bic= clustered_.begin();bic!=clustered_.end();bic++){
    reco::BasicClusterCollection bl = bic->second;
    for (int j=0;j<int(bl.size());++j){
      basicClusters.push_back(bl[j]);
    }
  }
  
  //Yay more sorting.
  sort(basicClusters.rbegin(), basicClusters.rend(), ClusterEtLess() );
  //Done!
  if ( debugLevel_ == pDEBUG )
    std::cout << "returning to producer. " << std::endl;
}


void HybridClusterAlgo::mainSearch(const EcalRecHitCollection* hits, const CaloSubdetectorGeometry*geometry)
{

	if ( debugLevel_ ==pDEBUG ) {
		std::cout << "HybridClusterAlgo Algorithm - looking for clusters" << std::endl;
		std::cout << "Found the following clusters:" << std::endl;
	}

	// Loop over seeds:
	std::vector<EcalRecHit>::iterator it;
	int clustercounter=0;

	for (it = seeds.begin(); it != seeds.end(); it++){
		std::vector <reco::BasicCluster> thisseedClusters;
		DetId itID = it->id();

		// make sure the current seed has not been used/will not be used in the future:
		std::set<DetId>::iterator seed_in_rechits_it = useddetids.find(itID);

		if (seed_in_rechits_it != useddetids.end()) continue;
		//If this seed is already used, then don't use it again.

		// output some info on the hit:
		if ( debugLevel_ == pDEBUG ){
			std::cout << "*****************************************************" << std::endl;
			std::cout << "Seed of energy E = " << it->energy() << " @ " << EBDetId(itID) 
				<< std::endl;
			std::cout << "*****************************************************" << std::endl;
		}

		//Make a navigator, and set it to the seed cell.
		EcalBarrelNavigator navigator(itID, topo_);

		//Now use the navigator to start building dominoes.

		//Walking positive in phi:
		std::vector <double> dominoEnergyPhiPlus;  //Here I will store the results of the domino sums
		std::vector <std::vector <EcalRecHit> > dominoCellsPhiPlus; //These are the actual EcalRecHit for dominos.

		//Walking negative in phi
		std::vector <double> dominoEnergyPhiMinus;  //Here I will store the results of the domino sums
		std::vector <std::vector <EcalRecHit> > dominoCellsPhiMinus; //These are the actual EcalRecHit for dominos.

		//The two sets together.
		std::vector <double> dominoEnergy;  //Here I will store the results of the domino sums
		std::vector <std::vector <EcalRecHit> > dominoCells; //These are the actual EcalRecHit for dominos.

		//First, the domino about the seed:
		std::vector <EcalRecHit> initialdomino;
		double e_init = makeDomino(navigator, initialdomino);
        if (e_init < Eseed) continue;

		if ( debugLevel_ == pDEBUG )
		{
			std::cout << "Made initial domino" << std::endl;
		}

		//
		// compute the phi road length
		double phiSteps;
		if (dynamicPhiRoad_ && e_init > 0)
		{
			double et5x5 = et25(navigator, hits, geometry);
			phiSteps = phiRoadAlgo_->barrelPhiRoad(et5x5);
			navigator.home();
		} else phiSteps = phiSteps_;

		//Positive phi steps.
		for (int i=0;i<phiSteps;++i){
			//remember, this always increments the current position of the navigator.
			DetId centerD = navigator.north();
			if (centerD.null())
				continue;
			if ( debugLevel_ == pDEBUG )
			{
				std::cout << "Step ++" << i << " @ " << EBDetId(centerD) << std::endl;
			}
			EcalBarrelNavigator dominoNav(centerD, topo_);

			//Go get the new domino.
			std::vector <EcalRecHit> dcells;
			double etemp = makeDomino(dominoNav, dcells);

			//save this information
			dominoEnergyPhiPlus.push_back(etemp);
			dominoCellsPhiPlus.push_back(dcells);
		}

		if ( debugLevel_ == pDEBUG )
			std::cout << "Got positive dominos" << std::endl;
		//return to initial position
		navigator.home();

		//Negative phi steps.
		for (int i=0;i<phiSteps;++i){
			//remember, this always decrements the current position of the navigator.
			DetId centerD = navigator.south();
			if (centerD.null())
				continue;

			if ( debugLevel_ == pDEBUG )
			{
				std::cout << "Step --" << i << " @ " << EBDetId(centerD) << std::endl;
			}
			EcalBarrelNavigator dominoNav(centerD, topo_);

			//Go get the new domino.
			std::vector <EcalRecHit> dcells;
			double etemp = makeDomino(dominoNav, dcells);

			//save this information
			dominoEnergyPhiMinus.push_back(etemp);
			dominoCellsPhiMinus.push_back(dcells);
		}

		if ( debugLevel_ == pDEBUG )
			std::cout << "Got negative dominos: " << std::endl;

		//Assemble this information:
		for (int i=int(dominoEnergyPhiMinus.size())-1;i >= 0;--i){
			dominoEnergy.push_back(dominoEnergyPhiMinus[i]);
			dominoCells.push_back(dominoCellsPhiMinus[i]);
		}
		dominoEnergy.push_back(e_init);
		dominoCells.push_back(initialdomino);
		for (int i=0;i<int(dominoEnergyPhiPlus.size());++i){
			dominoEnergy.push_back(dominoEnergyPhiPlus[i]);
			dominoCells.push_back(dominoCellsPhiPlus[i]);
		}

		//Ok, now I have all I need in order to go ahead and make clusters.
		if ( debugLevel_ == pDEBUG ){
			std::cout << "Dumping domino energies: " << std::endl;
			for (int i=0;i<int(dominoEnergy.size());++i){
				std::cout << "Domino: " << i << " E: " << dominoEnergy[i] << std::endl;
			}
		}


		//Identify the peaks in this set of dominos:
		//Peak==a domino whose energy is greater than the two adjacent dominos.
		//thus a peak in the local sense.
		std::vector <int> PeakIndex;
		for (int i=1;i<int(dominoEnergy.size())-1;++i){
			if (dominoEnergy[i] > dominoEnergy[i-1]
					&& dominoEnergy[i] >= dominoEnergy[i+1]
					&& dominoEnergy[i] > Eseed){
				PeakIndex.push_back(i);
			}
		}

		if ( debugLevel_ == pDEBUG )
			std::cout << "Found: " << PeakIndex.size() << " peaks." << std::endl;

		//Order these peaks by energy:
		for (int i=0;i<int(PeakIndex.size());++i){
			for (int j=0;j<int(PeakIndex.size())-1;++j){
				if (dominoEnergy[PeakIndex[j]] < dominoEnergy[PeakIndex[j+1]]){
					int ihold = PeakIndex[j+1];
					PeakIndex[j+1] = PeakIndex[j];
					PeakIndex[j] = ihold;
				}
			}
		}

		std::vector<int> OwnerShip;
		std::vector<double> LumpEnergy;
		for (int i=0;i<int(dominoEnergy.size());++i) OwnerShip.push_back(-1);

		//Loop over peaks.  
		double eThres = eThres_;
		double e5x5 = 0.0;
		for (int i = 0; i < int(PeakIndex.size()); ++i)
		{

			int idxPeak = PeakIndex[i];
			OwnerShip[idxPeak] = i;
			double lump = dominoEnergy[idxPeak];

			// compute eThres for this peak
			// if set to dynamic (otherwise uncanged from
			// fixed setting
			if (dynamicEThres_) {

				//std::cout << "i : " << i << " idxPeak " << idxPeak << std::endl;
				//std::cout << "    the dominoEnergy.size() = " << dominoEnergy.size() << std::endl;
				// compute e5x5 for this seed crystal
				//std::cout << "idxPeak, phiSteps " << idxPeak << ", " << phiSteps << std::endl;
				e5x5 = lump;
				//std::cout << "lump " << e5x5 << std::endl;
				if ((idxPeak + 1) < (int)dominoEnergy.size()) e5x5 += dominoEnergy[idxPeak + 1];
				//std::cout << "+1 " << e5x5 << std::endl;
				if ((idxPeak + 2) < (int)dominoEnergy.size()) e5x5 += dominoEnergy[idxPeak + 2];
				//std::cout << "+2 " << e5x5 << std::endl;
				if ((idxPeak - 1) > 0) e5x5 += dominoEnergy[idxPeak - 1];
				//std::cout << "-1 " << e5x5 << std::endl;
				if ((idxPeak - 2) > 0) e5x5 += dominoEnergy[idxPeak - 2];
				//std::cout << "-2 " << e5x5 << std::endl;
				// compute eThres
				eThres = (eThresA_ * e5x5) + eThresB_;   
				//std::cout << eThres << std::endl;
				//std::cout << std::endl;
			}

			//Loop over adjacent dominos at higher phi
			for (int j=idxPeak+1;j<int(dominoEnergy.size());++j){
				if (OwnerShip[j]==-1 && 
						dominoEnergy[j] > eThres
						&& dominoEnergy[j] <= dominoEnergy[j-1]){
					OwnerShip[j]= i;
					lump+=dominoEnergy[j];
				}
				else{
					break;
				}
			}
			//loop over adjacent dominos at lower phi.  Sum up energy of lumps.
			for (int j=idxPeak-1;j>=0;--j){
				if (OwnerShip[j]==-1 && 
						dominoEnergy[j] > eThres
						&& dominoEnergy[j] <= dominoEnergy[j+1]){
					OwnerShip[j]= i;
					lump+=dominoEnergy[j];
				}
				else{
					break;
				}
			}
			LumpEnergy.push_back(lump);
		}

		//Make the basic clusters:
		for (int i=0;i<int(PeakIndex.size());++i){
			bool HasSeedCrystal = false;
			//One cluster for each peak.
			std::vector<EcalRecHit> recHits;
			std::vector< std::pair<DetId, float> > dets;
			int nhits=0;
			for (int j=0;j<int(dominoEnergy.size());++j){	
				if (OwnerShip[j] == i){
					std::vector <EcalRecHit> temp = dominoCells[j];
					for (int k=0;k<int(temp.size());++k){
						dets.push_back( std::pair<DetId, float>(temp[k].id(), 1.) ); // by default energy fractions are 1
						if (temp[k].id()==itID)
							HasSeedCrystal = true;
						recHits.push_back(temp[k]);
						nhits++;
					}
				}  
			}
			if ( debugLevel_ == pDEBUG ){
				std::cout << "Adding a cluster with: " << nhits << std::endl;
				std::cout << "total E: " << LumpEnergy[i] << std::endl;
				std::cout << "total dets: " << dets.size() << std::endl;
			}

			//Get Calorimeter position
			Point pos = posCalculator_.Calculate_Location(dets,hits,geometry);

			//double totChi2=0;
			//double totE=0;
			std::vector<std::pair<DetId, float> > usedHits;
			for (int blarg=0;blarg<int(recHits.size());++blarg){
				//totChi2 +=0;
				//totE+=recHits[blarg].energy();
				usedHits.push_back(std::make_pair<DetId, float>(recHits[blarg].id(), 1.0));
				useddetids.insert(recHits[blarg].id());
			}

			//if (totE>0)
			//totChi2/=totE;

			if (HasSeedCrystal) {
				// note that this "basiccluster" has the seed crystal of the hyrbid, so record it
				seedClus_.push_back(reco::BasicCluster(LumpEnergy[i], pos, 
					reco::CaloID(reco::CaloID::DET_ECAL_BARREL), usedHits, 
					reco::CaloCluster::hybrid, itID));
				// and also add to the vector of clusters that will be used in constructing
				// the supercluster
                                thisseedClusters.push_back(reco::BasicCluster(LumpEnergy[i], pos,                                                               reco::CaloID(reco::CaloID::DET_ECAL_BARREL), 
                                     usedHits, reco::CaloCluster::hybrid, itID));
			}
			else {
				// note that if this "basiccluster" is not the one that seeded the hybrid, 
				// the seed crystal is unset in this entry in the vector of clusters that will
				// be used in constructing the super cluster
				thisseedClusters.push_back(reco::BasicCluster(LumpEnergy[i], pos,                                         			reco::CaloID(reco::CaloID::DET_ECAL_BARREL), 
				     usedHits, reco::CaloCluster::hybrid));
			}
		}


		// Make association so that superclusters can be made later.
		// but only if some BasicClusters have been found...
		if (thisseedClusters.size() > 0) 
		{
			clustered_.insert(std::make_pair(clustercounter, thisseedClusters));
			clustercounter++;
		}
	}//Seed loop

}

reco::SuperClusterCollection HybridClusterAlgo::makeSuperClusters(const reco::CaloClusterPtrVector& clustersCollection)
{
	//Here's what we'll return.
	reco::SuperClusterCollection SCcoll;

	//Here's our map iterator that gives us the appropriate association.
	std::map<int, reco::BasicClusterCollection>::iterator mapit;
	for (mapit = clustered_.begin();mapit!=clustered_.end();mapit++){

		reco::CaloClusterPtrVector thissc;
		reco::CaloClusterPtr seed;//This is not really a seed, but I need to tell SuperCluster something.
		//So I choose the highest energy basiccluster in the SuperCluster.

		std::vector <reco::BasicCluster> thiscoll = mapit->second; //This is the set of BasicClusters in this
		//SuperCluster

		double ClusterE = 0; //Sum of cluster energies for supercluster.
		//Holders for position of this supercluster.
		double posX=0;
		double posY=0;
		double posZ=0;

		//Loop over this set of basic clusters, find their references, and add them to the
		//supercluster.  This could be somehow more efficient.

		for (int i=0;i<int(thiscoll.size());++i){
			reco::BasicCluster thisclus = thiscoll[i]; //The Cluster in question.
			for (int j=0;j<int(clustersCollection.size());++j){
				//Find the appropriate cluster from the list of references
				reco::BasicCluster cluster_p = *clustersCollection[j];
				if (thisclus== cluster_p){ //Comparison based on energy right now.
					thissc.push_back(clustersCollection[j]);
					bool isSeed = false;
					for (int qu=0;qu<int(seedClus_.size());++qu){
						if (cluster_p == seedClus_[qu])
							isSeed = true;
					}
					if (isSeed) seed = clustersCollection[j];

					ClusterE += cluster_p.energy();
					posX += cluster_p.energy() * cluster_p.position().X();
					posY += cluster_p.energy() * cluster_p.position().Y();
					posZ += cluster_p.energy() * cluster_p.position().Z();

				}
			}//End loop over finding references.
		}//End loop over clusters.

		posX /= ClusterE;
		posY /= ClusterE;
		posZ /= ClusterE;

		/* //This part is moved to EgammaSCEnergyCorrectionAlgo
		   double preshowerE = 0.;
		   double phiWidth = 0.;
		   double etaWidth = 0.;
		//Calculate phiWidth & etaWidth for SuperClusters
		reco::SuperCluster suCl(ClusterE, math::XYZPoint(posX, posY, posZ), seed, thissc, preshowerE, phiWidth, etaWidth);
		SCShape_->Calculate_Covariances(suCl);
		phiWidth = SCShape_->phiWidth();
		etaWidth = SCShape_->etaWidth();
		//Assign phiWidth & etaWidth to SuperCluster as data members
		suCl.setPhiWidth(phiWidth);
		suCl.setEtaWidth(etaWidth);
		 */

		reco::SuperCluster suCl(ClusterE, math::XYZPoint(posX, posY, posZ), seed, thissc);
		SCcoll.push_back(suCl);

		if ( debugLevel_ == pDEBUG ){
			std::cout << "Super cluster sum: " << ClusterE << std::endl;
			std::cout << "Made supercluster with energy E: " << suCl.energy() << std::endl;
		}
	}//end loop over map
	sort(SCcoll.rbegin(), SCcoll.rend(), ClusterEtLess());
	return SCcoll;
}

double HybridClusterAlgo::makeDomino(EcalBarrelNavigator &navigator, std::vector <EcalRecHit> &cells)
{
	//At the beginning of this function, the navigator starts at the middle of the domino,
	//and that's where EcalBarrelNavigator::home() should send it.
	//Walk one crystal in eta to either side of the initial point.  Sum the three cell energies.
	//If the resultant energy is larger than Ewing, then go an additional cell to either side.
	//Returns:  Total domino energy.  Also, stores the cells used to create domino in the vector.
	cells.clear();
	double Etot = 0;

	//Ready?  Get the starting cell.
	DetId center = navigator.pos();
	EcalRecHitCollection::const_iterator center_it = recHits_->find(center);

	if (center_it!=recHits_->end()){
		EcalRecHit SeedHit = *center_it;
		if (useddetids.find(center) == useddetids.end() && excludedCrys_.find(center)==excludedCrys_.end()){ 
			Etot += SeedHit.energy();
			cells.push_back(SeedHit);
		}
	}
	//One step upwards in Ieta:
	DetId ieta1 = navigator.west();
	EcalRecHitCollection::const_iterator eta1_it = recHits_->find(ieta1);
	if (eta1_it !=recHits_->end()){
		EcalRecHit UpEta = *eta1_it;
		if (useddetids.find(ieta1) == useddetids.end() && excludedCrys_.find(ieta1)==excludedCrys_.end()){
			Etot+=UpEta.energy();
			cells.push_back(UpEta);
		}
	}

	//Go back to the middle.
	navigator.home();

	//One step downwards in Ieta:
	DetId ieta2 = navigator.east();
	EcalRecHitCollection::const_iterator eta2_it = recHits_->find(ieta2);
	if (eta2_it !=recHits_->end()){
		EcalRecHit DownEta = *eta2_it;
		if (useddetids.find(ieta2)==useddetids.end() && excludedCrys_.find(ieta2)==excludedCrys_.end()){
			Etot+=DownEta.energy();
			cells.push_back(DownEta);
		}
	}

	//Now check the energy.  If smaller than Ewing, then we're done.  If greater than Ewing, we have to
	//add two additional cells, the 'wings'
	if (Etot < Ewing) {
		navigator.home(); //Needed even here!!
		return Etot;  //Done!  Not adding 'wings'.
	}

	//Add the extra 'wing' cells.  Remember, we haven't sent the navigator home,
	//we're still on the DownEta cell.
	if (ieta2 != DetId(0)){
		DetId ieta3 = navigator.east(); //Take another step downward.
		EcalRecHitCollection::const_iterator eta3_it = recHits_->find(ieta3);
		if (eta3_it != recHits_->end()){
			EcalRecHit DownEta2 = *eta3_it;
			if (useddetids.find(ieta3)==useddetids.end() && excludedCrys_.find(ieta3)==excludedCrys_.end()){
				Etot+=DownEta2.energy();
				cells.push_back(DownEta2);
			}
		}
	}
	//Now send the navigator home.
	navigator.home();
	navigator.west(); //Now you're on eta1_it
	if (ieta1 != DetId(0)){
		DetId ieta4 = navigator.west(); //Take another step upward.
		EcalRecHitCollection::const_iterator eta4_it = recHits_->find(ieta4);
		if (eta4_it != recHits_->end()){
			EcalRecHit UpEta2 = *eta4_it;
			if (useddetids.find(ieta4) == useddetids.end() && excludedCrys_.find(ieta4)==excludedCrys_.end()){
				Etot+=UpEta2.energy();
				cells.push_back(UpEta2);
			}
		}
	}
	navigator.home();
	return Etot;
}

double HybridClusterAlgo::et25(EcalBarrelNavigator &navigator, 
		const EcalRecHitCollection *hits, 
		const CaloSubdetectorGeometry *geometry)
{

	DetId thisDet;
	std::vector< std::pair<DetId, float> > dets;
	dets.clear();
	EcalRecHitCollection::const_iterator hit;
	double energySum = 0.0;

	for (int dx = -2; dx < 3; ++dx)
	{
		for (int dy = -2; dy < 3; ++ dy)
		{
			//std::cout << "dx, dy " << dx << ", " << dy << std::endl;
			thisDet = navigator.offsetBy(dx, dy);
			navigator.home();

			if (thisDet != DetId(0))
			{
				hit = recHits_->find(thisDet);
				if (hit != recHits_->end()) 
				{
					dets.push_back( std::pair<DetId, float>(thisDet, 1.) ); // by default hit energy fraction is set to 1
					energySum += hit->energy();
				}
			}
		}
	}

	// convert it to ET
	//std::cout << "dets.size(), energySum: " << dets.size() << ", " << energySum << std::endl;
	Point pos = posCalculator_.Calculate_Location(dets, hits, geometry);
	double et = energySum/cosh(pos.eta());
	return et;

}

