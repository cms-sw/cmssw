#include "RecoEcal/EcalClusterAlgos/interface/HybridClusterAlgo.h"
#include "RecoEcal/EcalClusterAlgos/interface/PositionAwareHit.h"

#include "RecoCaloTools/Navigation/interface/EBDetIdNavigator.h"

#include <iostream>
#include <map>
#include <vector>

void HybridClusterAlgo::mainSearch(const CaloSubdetectorGeometry & geometry)
{
  std::cout << "HybridClusterAlgo Algorithm - looking for clusters" << std::endl;
  std::cout << "Found the following clusters:" << std::endl;
  
  // Loop over seeds:
  std::vector<PositionAwareHit>::iterator it;
  for (it = seeds.begin(); it != seeds.end(); it++){
    // make sure the current seed has not been used/will not be used in the future:
    std::map<EBDetId, PositionAwareHit>::iterator seed_in_rechits_it = rechits_m.find(it->getId());
    if (seed_in_rechits_it->second.isUsed()) continue;
    
//     seed_in_rechits_it->second.use();
    
    // output some info on the hit:
    std::cout << "*****************************************************" << std::endl;
    std::cout << "Seed of energy E = " << it->getEnergy() 
	      << ", eta = " << it->getEta() 
	      << ", phi = " << it->getPhi() 
	      << std::endl;
    std::cout << "*****************************************************" << std::endl;
    //    std::cout << "Included RecHits:" << std::endl;      
    
    
    EBDetIdNavigator navigator(it->getId());
    //Now use the navigator to start building dominoes.
    
    std::vector <double> dominoEnergyPhiPlus;  //Here I will store the results of the domino sums
    std::vector <std::vector <PositionAwareHit> > dominoCellsPhiPlus; //These are the actual PositionAwareHits for dominos.
    
    std::vector <double> dominoEnergyPhiMinus;  //Here I will store the results of the domino sums
    std::vector <std::vector <PositionAwareHit> > dominoCellsPhiMinus; //These are the actual PositionAwareHits for dominos.
    
    std::vector <double> dominoEnergy;  //Here I will store the results of the domino sums
    std::vector <std::vector <PositionAwareHit> > dominoCells; //These are the actual PositionAwareHits for dominos.
    
    //First, the domino about the seed:
    std::vector <PositionAwareHit> initialdomino;
    double e_init = makeDomino(navigator, initialdomino);
    
    //Positive phi steps.
    for (int i=0;i<phi_steps;++i){
      //remember, this always increments the current position of the navigator.
      EBDetId centerD = navigator.incrementIphi();
      EBDetIdNavigator dominoNav(centerD);
      
      //Go get the new domino.
      std::vector <PositionAwareHit> dcells;
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
      EBDetId centerD = navigator.decrementIphi();
      EBDetIdNavigator dominoNav(centerD);
      
      //Go get the new domino.
      std::vector <PositionAwareHit> dcells;
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
	  std::vector <PositionAwareHit> temp = dominoCells[j];
	  for (int k=0;k<int(temp.size());++k){
	    reco::EcalRecHitData data(temp[k].getEnergy(),0,temp[k].getId());
	    recHits.push_back(data);
	    nhits++;
	  }
	}  
      }
      std::cout << "Adding a cluster with: " << nhits << std::endl;
      std::cout << "total E: " << LumpEnergy[i] << std::endl;
      //Get Calorimeter position
      Point pos = getECALposition(recHits, geometry);
      clusters.push_back(reco::BasicCluster(recHits, 100, pos));
    }
  }
}

double HybridClusterAlgo::makeDomino(EBDetIdNavigator &navigator, std::vector <PositionAwareHit> &cells)
{
  //At the beginning of this function, the navigator starts at the middle of the domino,
  //and that's where EBDetIdNavigator::home() should send it.
  //Walk one crystal in eta to either side of the initial point.  Sum the three cell energies.
  //If the resultant energy is larger than Ewing, then go an additional cell to either side.
  //Returns:  Total domino energy.  Also, stores the cells used to create domino in the vector.
  cells.clear();
  double Etot = 0;

  //Ready?  Get the starting cell.
  EBDetId center = navigator.pos();
  std::map<EBDetId, PositionAwareHit>::iterator center_it = rechits_m.find(center);

  if (center_it==rechits_m.end()) return 0; //Didn't find that ID.
  if (center_it->second.isUsed()) return 0; //Already used that ID.  Terminate either way.

  Etot += center_it->second.getEnergy();
  cells.push_back(center_it->second);
  center_it->second.use();

  //One step upwards in Ieta:
  EBDetId ieta1 = navigator.incrementIeta();
  std::map<EBDetId, PositionAwareHit>::iterator eta1_it = rechits_m.find(ieta1);
  if (eta1_it !=rechits_m.end()){
    if (!eta1_it->second.isUsed()){
      Etot+=eta1_it->second.getEnergy();
      cells.push_back(eta1_it->second);
      eta1_it->second.use();
    }
  }

  //Go back to the middle.
  navigator.home();

  //One step downwards in Ieta:
  EBDetId ieta2 = navigator.decrementIeta();
  std::map<EBDetId, PositionAwareHit>::iterator eta2_it = rechits_m.find(ieta2);
  if (eta2_it !=rechits_m.end()){
    if (!eta2_it->second.isUsed()){
      Etot+=eta2_it->second.getEnergy();
      cells.push_back(eta2_it->second);
      eta2_it->second.use();
    }
  }

  //Now check the energy.  If smaller than Ewing, then we're done.  If greater than Ewing, we have to
  //add two additional cells, the 'wings'
  if (Etot < Ewing) return Etot;  //Done!  Not adding 'wings'.

  if (eta2_it !=rechits_m.end() && !eta2_it->second.isUsed()){
    EBDetId ieta3 = navigator.decrementIeta(); //Take another step downward.
    std::map<EBDetId, PositionAwareHit>::iterator eta3_it = rechits_m.find(ieta3);
    if (eta3_it != rechits_m.end()){
      if (!eta3_it->second.isUsed()){
	Etot+=eta3_it->second.getEnergy();
	cells.push_back(eta3_it->second);
	eta3_it->second.use();
      }
    }
  }

  navigator.home();
  if (eta1_it !=rechits_m.end() && !eta1_it->second.isUsed()){
    navigator.incrementIeta();
    EBDetId ieta4 = navigator.incrementIeta(); //Take another step upward.
    std::map<EBDetId, PositionAwareHit>::iterator eta4_it = rechits_m.find(ieta4);
    if (eta4_it != rechits_m.end()){
      if (!eta4_it->second.isUsed()){
	Etot+=eta4_it->second.getEnergy();
	cells.push_back(eta4_it->second);
	eta4_it->second.use();
      }
    }
  }
  navigator.home();
  return Etot;
}

