// -*- C++ -*-
//
// Package:    Castor
// Class:      Castor
// 
/**\class Castor KtAlgorithm.cc RecoLocalCalo/Castor/src/KtAlgorithm.cc

 Description: KtAlgorithm implementation for use with the CastorProducers.
 	      Code is based on the original KtJet 1.08 package source code made by 
	      J. Butterworth, J. Couchman, B. Cox & B. Waugh and adapted to work with 
	      CastorTower objects as input and CastorJets as output.

 Implementation:
      
*/
//
// Original Author:  Hans Van Haevermaet
//         Created:  Sat May 24 12:00:56 CET 2008
// $Id: KtAlgorithm.cc,v 1.1.2.1 2008/08/30 20:46:32 hvanhaev Exp $
//
//

// includes
#include <iostream>
#include <algorithm>
#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorJet.h"
#include "RecoLocalCalo/Castor/interface/KtAlgorithm.h"

// namespaces
using namespace std;
using namespace reco;
using namespace math;

// typedefs
typedef math::XYZPointD Point;
typedef ROOT::Math::RhoEtaPhiPoint TowerPoint;

// help function to calculate phi within [-pi,+pi]
double KtAlgorithm::phiangle (double testphi) {
  double phi = testphi;
  while (phi>M_PI) phi -= (2*M_PI);
  while (phi<-M_PI) phi += (2*M_PI);
  return phi;
}

// recombination scheme function to make new jet
CastorJet KtAlgorithm::calcRecom (CastorJet a, CastorJet b, int recom) {
  double newE, newEmE, newHadE, newRatio, newPhi, deltaPhi, newWidth, newDepth;
  newE = a.energy() + b.energy();
  newEmE = a.emEnergy() + b.emEnergy();
  newHadE = a.hadEnergy() + b.hadEnergy();
  deltaPhi = phiangle(b.phi() - a.phi());
  if (recom == 2) {
  	// use recombination scheme 2 = Pt
	newPhi = a.phi() + deltaPhi*b.energy()/newE;
  } else if (recom == 3) {
  	// use recombination scheme 3 = Pt2
        newPhi = a.phi() + deltaPhi*(b.energy()*b.energy())/((a.energy()*a.energy()) + (b.energy()*b.energy()));
  } else {
  	// error
	newPhi = 0.;
	cout << "You are using a wrong recombination scheme. Check the input tag, this should be 2(pt) or 3(pt2). \n";
  } 
  newPhi = phiangle(newPhi);
  TowerPoint temp(1.,a.eta(),newPhi);
  Point position(temp);
  newRatio = newEmE/newE;
  newWidth = a.width() + b.width();
  newDepth = (a.depth()*a.energy() + b.depth()*b.energy())/newE; // take weighted depth average
  CastorTowerCollection newUsedTowers;
  CastorTowerCollection aTowers = a.getUsedTowers();
  CastorTowerCollection bTowers = b.getUsedTowers();
  aTowers.insert(aTowers.end(),bTowers.begin(),bTowers.end());
  newUsedTowers = aTowers;
  CastorJet newJet(newE,position,newEmE,newHadE,newRatio,newWidth,newDepth,newUsedTowers);
  return newJet;
}

// function to calculate distance delta R between 2 jets
double KtAlgorithm::calcDistanceDeltaR (CastorJet a, CastorJet b) {
  double rsq,esq,asq,bsq,kt,deltaPhi;
  deltaPhi = phiangle(a.phi() - b.phi());
  rsq = deltaPhi*deltaPhi;
  asq = a.energy()*a.energy();
  bsq = b.energy()*b.energy();
  esq = std::min(asq,bsq);
  kt = esq*rsq;
  return kt;
}

// calculate dPairs vectors
std::vector<std::vector<double> > KtAlgorithm::calcdPairs (CastorJetCollection protojets, std::vector<std::vector<double> > dPairs) {
  int nRows = protojets.size();
  dPairs.reserve(nRows);
  // fill dPairs vector with distances
  for (int i=0;i<nRows-1;++i) {
  	std::vector<double> tempvec;
	tempvec.reserve(nRows);
  	for (int j=i+1;j<nRows;++j) {
		double D = calcDistanceDeltaR(protojets[i],protojets[j]);
		tempvec.push_back(D);	
	}
	dPairs.push_back(tempvec);
  }
  return dPairs;
}

// calculate ddi vectors
std::vector<double> KtAlgorithm::calcddi (CastorJetCollection protojets, std::vector<double> ddi) {  
  int nRows = protojets.size();
  ddi.reserve(nRows);
  // fill ddi vector with energy squared
  for (int i=0;i<nRows;++i) {
  	ddi.push_back(protojets[i].energy()*protojets[i].energy());
  }
  return ddi;
}

// main public function executes Kt algorithm, is called to give results      
CastorJetCollection KtAlgorithm::runKtAlgo (const CastorTowerCollection inputtowers, const int recom, const double rParameter) {
  
  // get and check input size
  int nTowers = inputtowers.size();
  if (nTowers==0) {
  	cout << "Warning: You are trying to run the KtAlgorithm with 0 input towers. \n";
  }
  
  // define output
  CastorJetCollection protojets, jets;
  protojets.reserve(inputtowers.size());
  // copy input to working vector
  CastorTowerCollection towers = inputtowers;
  
  // copy towers to protojets vector
  for (size_t i=0;i<towers.size();i++) {
  	TowerPoint temp(1.,towers[i].eta(),towers[i].phi());
	Point position(temp);
	CastorTowerCollection usedTowers;
	usedTowers.push_back(towers[i]);
  	protojets.push_back(CastorJet(towers[i].energy(),position,towers[i].emEnergy(),towers[i].hadEnergy(),towers[i].emtotRatio(),
	towers[i].width(),towers[i].depth(),usedTowers));
  } 
  
  // start merging until only one jet is left
  int njet;
  while (njet=protojets.size()>1) {
         
        // call calcddi and calcdPairs function
  	std::vector<double> ddi;
  	std::vector<std::vector<double> > dPairs;
  	ddi = calcddi(protojets,ddi);
	dPairs = calcdPairs(protojets,dPairs);
  
  	int iPairMin, jPairMin;
	double dPairMin, dJetMin;
	
	// find min of dPairs
	double temp = dPairs[0][0];
	iPairMin = 0;
	jPairMin = 1;
	for (size_t i=0;i<dPairs.size();i++) {
	     for (size_t j=0;j<dPairs[i].size();j++) { 
	     	  if (dPairs[i][j] < temp) {
		  	temp = dPairs[i][j];
			iPairMin = i;
			jPairMin = j+i+1;
		  } 
	     }
	}
	dPairMin = temp;
	
	// find min of ddi vector
	double temp2 = ddi[0];
	int iJetMin = 0;
	for (size_t i=0;i<ddi.size();i++) {
		if (ddi[i] < temp2) {
			temp2 = ddi[i];
			iJetMin = i;
		}
	}
	dJetMin = temp2;
	dJetMin = dJetMin*rParameter*rParameter;
	
	// take min dPairs and ddi and merge or not
	if ( dPairMin < dJetMin ) {
		// merge the 2 protojets and put it in the protojet list at iPairMin
		CastorJet recombined = calcRecom(protojets[iPairMin],protojets[jPairMin],recom);
  		protojets[iPairMin] = recombined;
  		protojets.erase(protojets.begin()+jPairMin);
	} else {
		// put protojet iJetMin in list of final jets and remove protojet iJetMin
		jets.push_back(protojets[iJetMin]);
		protojets.erase(protojets.begin()+iJetMin);
	}
	
  }
  
  // end of loop, should have njet = 1
  jets.push_back(protojets[0]); // if there's only one protojet left, make a jet of it, is this ok?
  
  // try to calculate width of all the jets
  for (size_t i=0;i<jets.size();i++) {
  	std::vector<CastorTower> usedTowers = jets[i].getUsedTowers();
	double sum_e = 0.;
	double sum_distances = 0.;
	double jetwidth;
	
	//cout << " calculate width of jet " << i << "\n";
	for (size_t j=0;j<usedTowers.size();j++) {
		sum_e = sum_e + usedTowers[j].energy();
		sum_distances = sum_distances + pow(phiangle(usedTowers[j].phi() - jets[i].phi()),2)*usedTowers[j].energy();
		//cout << " tower " << j << " is at " << usedTowers[j].phi() << " and has " << usedTowers[j].energy() << " GeV energy \n";
	}
	
	jetwidth = sqrt(sum_distances/sum_e);
	//cout << "there are " << usedTowers.size() << " used towers and the jet has width " << jetwidth << "\n";
	
	TowerPoint jettemppos(1.0,jets[i].eta(),jets[i].phi());
	Point jetpos(jettemppos);
	jets[i] = CastorJet(jets[i].energy(),jetpos,jets[i].emEnergy(),jets[i].hadEnergy(),jets[i].emtotRatio(),jetwidth,jets[i].depth(),jets[i].getUsedTowers());
  }
  
  
  return jets;
}

