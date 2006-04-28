#include "RecoEcal/EgammaClusterAlgos/interface/HybridClusterAlgo.h"
#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelHardcodedTopology.h"
#include <iostream>
#include <map>
#include <vector>

// Return a vector of clusters from a collection of EcalRecHits:
void HybridClusterAlgo::makeClusters(EcalRecHitCollection & rechits, edm::ESHandle<CaloGeometry> geometry_h,  reco::BasicClusterCollection &basicClusters)
{
  //Clear the vectors:
  //map that keeps track of used det hits
  rechits_m.clear();

  //vector of seeds
  seeds.clear();

  //map of supercluster/basiccluster association
  _clustered.clear();
  
  std::cout << "Cleared vectors, starting clusterization..." << std::endl;
  std::cout << "Number of RecHits in event = " << rechits.size() << std::endl;
  const CaloSubdetectorGeometry *geometry_p = (*geometry_h).getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  CaloSubdetectorGeometry const geometry = *geometry_p;
  EcalRecHitCollection::iterator it;
  for (it = rechits.begin(); it != rechits.end(); it++){
    //Double purpose loop:
    //Make the map of DetID, <EcalRecHit,used> pairs
    //Make the vector of seeds that we're going to use.
    //One of the few places position is used, needed for ET calculation.    
    const CaloCellGeometry *this_cell = geometry.getGeometry(it->id());
    GlobalPoint position = this_cell->getPosition();
    
    std::pair<EcalRecHit, bool> HitBool = std::make_pair(*it, false);
    rechits_m.insert(std::make_pair(it->id(), HitBool));    

    float ET = it->energy() * sin(position.theta());
    // and high ET hits are also seeds:
    if (ET > eb_st){
      seeds.push_back(*it);
      std::cout << "Seed ET: " << ET << std::endl;
      std::cout << "Seed E: " << it->energy() << std::endl;
    }
  }
  
  //Yay sorting.
  std::cout << "Built vector of seeds, about to sort them...";

  //Needs three argument sort with seed comparison operator
  sort(seeds.begin(), seeds.end(), less_mag());
  std::cout << "done" << std::endl;

  //Now to do the work.
  std::cout << "About to call mainSearch...";
 
  mainSearch(geometry);
  std::cout << "done" << std::endl;
  std::map<int, reco::BasicClusterCollection>::iterator bic; 
  for (bic= _clustered.begin();bic!=_clustered.end();bic++){
    reco::BasicClusterCollection bl = bic->second;
    for (int j=0;j<int(bl.size());++j){
      basicClusters.push_back(bl[j]);
    }
  }

  //Yay more sorting.
  sort(basicClusters.begin(), basicClusters.end());
  //Done!
}

reco::SuperClusterCollection HybridClusterAlgo::makeSuperClusters(reco::BasicClusterRefVector clustersCollection)
{
  //Here's what we'll return.
  reco::SuperClusterCollection SCcoll;

  //Here's our map iterator that gives us the appropriate association.
  std::map<int, reco::BasicClusterCollection>::iterator mapit;
  for (mapit = _clustered.begin();mapit!=_clustered.end();mapit++){
    //    reco::SuperCluster thissc;
    reco::BasicClusterRefVector thissc;
    reco::BasicClusterRef seed;//This is not really a seed, but I need to tell SuperCluster something.
                               //So I choose the highest energy basiccluster in the SuperCluster.
    
    std::vector <reco::BasicCluster> thiscoll = mapit->second; //This is the set of BasicClusters in this
                                                               //SuperCluster
    double ClusterE =0; //Sum of cluster energies for supercluster.

    //Loop over this set of basic clusters, find their references, and add them to the
    //supercluster.  This could be somehow more efficient.
    for (int i=0;i<int(thiscoll.size());++i){
      reco::BasicCluster thisclus = thiscoll[i]; //The Cluster in question.
      for (int j=0;j<int(clustersCollection.size());++j){
	//Find the appropriate cluster from the list of references
	reco::BasicCluster cluster_p = *clustersCollection[j];
	if (thisclus== cluster_p){ //Comparison based on energy right now.
	  thissc.push_back(clustersCollection[j]);
	  if (i==0) seed = clustersCollection[j];
	  ClusterE += cluster_p.energy();
	}
      }//End loop over finding references.
    }//End loop over clusters.
    reco::SuperCluster suCl(ClusterE, (*seed).position(), seed, thissc);
    SCcoll.push_back(suCl);
    std::cout << "Super cluster sum: " << ClusterE << std::endl;
    std::cout << "Made supercluster with energy E: " << suCl.energy() << std::endl;
  }//end loop over map
  sort(SCcoll.begin(), SCcoll.end());
  return SCcoll;
}

void HybridClusterAlgo::mainSearch(const CaloSubdetectorGeometry geometry)
{
  std::cout << "HybridClusterAlgo Algorithm - looking for clusters" << std::endl;
  std::cout << "Found the following clusters:" << std::endl;

  // Loop over seeds:
  std::vector<EcalRecHit>::iterator it;
  int clustercounter=0;
  for (it = seeds.begin(); it != seeds.end(); it++){
    std::vector <reco::BasicCluster> thisseedClusters;
    EBDetId itID = it->id();
    // make sure the current seed has not been used/will not be used in the future:
    std::map<EBDetId, std::pair<EcalRecHit, bool> >::iterator seed_in_rechits_it = rechits_m.find(itID);
    std::pair <EcalRecHit, bool> thisSeed = seed_in_rechits_it->second;
    //If this seed is already used, then don't use it again.
    if (thisSeed.second) continue;
    
    // output some info on the hit:
    std::cout << "*****************************************************" << std::endl;
    std::cout << "Seed of energy E = " << thisSeed.first.energy() 
	      << std::endl;
    std::cout << "*****************************************************" << std::endl;

    //Make a navigator, and set it to the seed cell.
    EcalBarrelHardcodedTopology *topo = new EcalBarrelHardcodedTopology();
    EcalBarrelNavigator navigator(itID, topo);

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
    
    //Positive phi steps.
    for (int i=0;i<phi_steps;++i){
      //remember, this always increments the current position of the navigator.
      EBDetId centerD = navigator.north();

      EcalBarrelNavigator dominoNav(centerD, topo);
      
      //Go get the new domino.
      std::vector <EcalRecHit> dcells;
      double etemp = makeDomino(dominoNav, dcells);
      
      //save this information
      dominoEnergyPhiPlus.push_back(etemp);
      dominoCellsPhiPlus.push_back(dcells);
    }
    std::cout << "Got positive dominos" << std::endl;
    //return to initial position
    navigator.home();
    
    //Negative phi steps.
    for (int i=0;i<phi_steps;++i){
      //remember, this always decrements the current position of the navigator.
      EBDetId centerD = navigator.south();
      EcalBarrelNavigator dominoNav(centerD, topo);
      
      //Go get the new domino.
      std::vector <EcalRecHit> dcells;
      double etemp = makeDomino(dominoNav, dcells);
      
      //save this information
      dominoEnergyPhiMinus.push_back(etemp);
      dominoCellsPhiMinus.push_back(dcells);
    }
    
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
    std::cout << "Dumping domino energies: " << std::endl;
    for (int i=0;i<int(dominoEnergy.size());++i){
      std::cout << "Domino: " << i << " E: " << dominoEnergy[i] << std::endl;
    }


    //Identify the peaks in this set of dominos:
    //Peak==a domino whose energy is greater than the two adjacent dominos.
    //thus a peak in the local sense.
    std::vector <int> PeakIndex;
    for (int i=1;i<int(dominoEnergy.size())-1;++i){
      if (dominoEnergy[i] > dominoEnergy[i-1]
	  && dominoEnergy[i] > dominoEnergy[i+1]
	  && dominoEnergy[i] > Eseed){
	PeakIndex.push_back(i);
      }
    }
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
    for (int i=0;i<int(PeakIndex.size());++i){
      int idxPeak = PeakIndex[i];
      OwnerShip[idxPeak] = i;
      double lump = dominoEnergy[idxPeak];
      //Loop over adjacent dominos at higher phi
      for (int j=idxPeak+1;j<int(dominoEnergy.size());++j){
	if (OwnerShip[j]==-1 && 
	    dominoEnergy[j] > Ethres
	    && dominoEnergy[j] < dominoEnergy[j-1]){
	  OwnerShip[j]= i;
	  lump+=dominoEnergy[j];
	}
	else{
	  break;
	}
      }
      //loop over adjacent dominos at lower phi
      for (int j=idxPeak-1;j>=0;--j){
	if (OwnerShip[j]==-1 && 
	    dominoEnergy[j] > Ethres
	    && dominoEnergy[j] < dominoEnergy[j+1]){
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
      //One cluster for each peak.
      std::vector<reco::EcalRecHitData> recHits;
      int nhits=0;
      for (int j=0;j<int(dominoEnergy.size());++j){	
	if (OwnerShip[j] == i){
	  std::vector <EcalRecHit> temp = dominoCells[j];
	  for (int k=0;k<int(temp.size());++k){
	    reco::EcalRecHitData data(temp[k].energy(),0,temp[k].id());
	    recHits.push_back(data);
	    nhits++;
	  }
	}  
      }
      std::cout << "Adding a cluster with: " << nhits << std::endl;
      std::cout << "total E: " << LumpEnergy[i] << std::endl;
      //Get Calorimeter position
      Point pos = getECALposition(recHits, geometry);
      
      double totChi2=0;
      double totE=0;
      for (int blarg=0;blarg<int(recHits.size());++blarg){
	totChi2 +=recHits[blarg].energy()*recHits[blarg].chi2();
	totE+=recHits[blarg].energy();
      }
      if (totE>0)
	totChi2/=totE;
      
      //thisseedClusters.push_back(reco::BasicCluster(recHits,1,pos));
      thisseedClusters.push_back(reco::BasicCluster(LumpEnergy[i],pos,totChi2));
    }
    _clustered.insert(std::make_pair(clustercounter, thisseedClusters));    
    clustercounter++;
  }//Seed loop
  
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
  EBDetId center = navigator.pos();
  std::map<EBDetId, std::pair<EcalRecHit, bool> >::iterator center_it = rechits_m.find(center);
  
  if (center_it==rechits_m.end()) return 0; //Didn't find that ID.
  std::pair <EcalRecHit, bool> SeedHit = center_it->second;
  if (SeedHit.second) return 0; //Already used that ID.  Terminate either way.

  Etot += SeedHit.first.energy();
  cells.push_back(SeedHit.first);
  //Mark cell as used.
  center_it->second.second = true;

  //One step upwards in Ieta:
  EBDetId ieta1 = navigator.west();
  std::map<EBDetId, std::pair<EcalRecHit, bool> >::iterator eta1_it = rechits_m.find(ieta1);
  if (eta1_it !=rechits_m.end()){
    std::pair <EcalRecHit, bool> UpEta = eta1_it->second;
    if (!UpEta.second){
      Etot+=UpEta.first.energy();
      cells.push_back(UpEta.first);
      //Mark cell as used.
      eta1_it->second.second = true;
    }
  }

  //Go back to the middle.
  navigator.home();

  //One step downwards in Ieta:
  EBDetId ieta2 = navigator.east();
  std::map<EBDetId, std::pair<EcalRecHit, bool> >::iterator eta2_it = rechits_m.find(ieta2);
  if (eta2_it !=rechits_m.end()){
    std::pair <EcalRecHit, bool> DownEta = eta2_it->second;
    if (!DownEta.second){
      Etot+=DownEta.first.energy();
      cells.push_back(DownEta.first);
      //Mark cell as used.
      eta2_it->second.second=true;
    }
  }

  //Now check the energy.  If smaller than Ewing, then we're done.  If greater than Ewing, we have to
  //add two additional cells, the 'wings'
  if (Etot < Ewing) return Etot;  //Done!  Not adding 'wings'.

  //Add the extra 'wing' cells.  Remember, we haven't sent the navigator home,
  //we're still on the DownEta cell.
  if (eta2_it !=rechits_m.end()){
    EBDetId ieta3 = navigator.east(); //Take another step downward.
    std::map<EBDetId, std::pair<EcalRecHit, bool> >::iterator eta3_it = rechits_m.find(ieta3);
    if (eta3_it != rechits_m.end()){
      std::pair <EcalRecHit, bool> DownEta2 = eta3_it->second;
      if (!DownEta2.second){
	Etot+=DownEta2.first.energy();
	cells.push_back(DownEta2.first);
	//Mark cell as used.
	eta3_it->second.second = true;
      }
    }
  }

  //Now send the navigator home.
  navigator.home();
  //Recall, eta1_it is the position incremented one time.
  if (eta1_it !=rechits_m.end()){
    navigator.west(); //Now you're on eta1_it
    EBDetId ieta4 = navigator.west(); //Take another step upward.
    std::map<EBDetId, std::pair<EcalRecHit, bool> >::iterator eta4_it = rechits_m.find(ieta4);
    if (eta4_it != rechits_m.end()){
      std::pair <EcalRecHit, bool> UpEta2 = eta4_it->second;
      if (!UpEta2.second){
	Etot+=UpEta2.first.energy();
	cells.push_back(UpEta2.first);
	eta4_it->second.second = true;
      }
    }
  }
  navigator.home();
  return Etot;
}

