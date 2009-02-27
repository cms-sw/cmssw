// -*- C++ -*-
//
// Package:    Castor
// Class:      KtAlgorithm
// 
/**\class KtAlgorithm KtAlgorithm.cc RecoLocalCalo/Castor/src/KtAlgorithm.cc

 Description: KtAlgorithm implementation for use with the CastorClusterProducer.
 	      Code is based on the original KtJet 1.08 package source code made by 
	      J. Butterworth, J. Couchman, B. Cox & B. Waugh and adapted to work with 
	      CastorTower objects as input and CastorCluster objects as output.
 Implementation:
      
*/
//
// Original Author:  Hans Van Haevermaet, Benoit Roland
//         Created:  Sat May 24 12:00:56 CET 2008
// $Id: KtAlgorithm.cc,v 1.2 2008/11/24 22:43:28 hvanhaev Exp $
//
//

// include
#include <iostream>
#include <algorithm>
#include <TMath.h>

#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorCluster.h"

#include "RecoLocalCalo/Castor/interface/KtAlgorithm.h"

#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

#define debug 0

// namespaces
using namespace std;
using namespace reco;
using namespace math;

// typedef
typedef math::XYZPointD Point;
typedef ROOT::Math::RhoEtaPhiPoint TowerPoint;

// help function to calculate phi within [-pi,+pi]
double KtAlgorithm::phiangle (double testphi) {
  double phi = testphi;
  while (phi>M_PI) phi -= (2*M_PI);
  while (phi<-M_PI) phi += (2*M_PI);
  return phi;
}

// recombination scheme function to make new cluster
CastorCluster KtAlgorithm::calcRecom (CastorCluster a, CastorCluster b, int recom) {
  double newE, newEmE, newHadE, newfem, newPhi, deltaPhi, deltaRho, newWidth, newDepth, newRho;

  double Ea = a.energy();
  double Eb = b.energy();

  newE = Ea + Eb;
  newEmE = a.emEnergy() + b.emEnergy();
  newHadE = a.hadEnergy() + b.hadEnergy();
  deltaPhi = phiangle(b.phi() - a.phi());
  deltaRho = b.rho() - a.rho();

  if (recom == 2) {
    // use recombination scheme 2 = Pt
    newPhi = a.phi() + deltaPhi*Eb/newE;
    newRho = a.rho() + deltaRho*Eb/newE; 
  } else if (recom == 3) {
    // use recombination scheme 3 = Pt2
    newPhi = a.phi() + deltaPhi*(Eb*Eb)/((Ea*Ea) + (Eb*Eb));
    newRho = a.rho() + deltaRho*(Eb*Eb)/((Ea*Ea) + (Eb*Eb));
  } else {
    // error
    newPhi = 0.;
    newRho = 0.;
    cout << "You are using a wrong recombination scheme. Check the input tag, this should be 2(pt) or 3(pt2). \n";
  } 

  if(debug) cout<<""<<endl;
  if(debug) cout<<"merging of tower"<<endl;
  if(debug) cout<<"tower a: "<<" energy: "<<Ea<<" rho: "<<a.rho()<<endl;
  if(debug) cout<<"tower b: "<<" energy: "<<Eb<<" rho: "<<b.rho()<<endl;
  if(debug) cout<<"tower new: "<<" energy: "<<newE<<" rho: "<<newRho<<endl;
  if(debug) getchar();

  newPhi = phiangle(newPhi);
  TowerPoint temp(newRho,a.eta(),newPhi);
  Point position(temp);
  newfem = newEmE/newE;
  newWidth = a.width() + b.width();
  newDepth = (a.depth()*Ea + b.depth()*Eb)/newE; // take weighted depth average

  CastorTowerRefVector aTowers = a.getUsedTowers();
  CastorTowerRefVector bTowers = b.getUsedTowers();
  CastorTowerRefVector newTowers;

  for (CastorTower_iterator it_aTower = (aTowers).begin(); it_aTower != (aTowers).end(); it_aTower++) {
    reco::CastorTowerRef aTower_p = *it_aTower;
    newTowers.push_back(aTower_p);
  }  

  for (CastorTower_iterator it_bTower = (bTowers).begin(); it_bTower != (bTowers).end(); it_bTower++) {
    reco::CastorTowerRef bTower_p = *it_bTower;
    newTowers.push_back(bTower_p);
  }

  CastorCluster newCluster(newE,position,newEmE,newHadE,newfem,newWidth,newDepth,0,0,newTowers); 
  return newCluster;
}  

// function to calculate distance delta R between 2 clusters
double KtAlgorithm::calcDistanceDeltaR (CastorCluster a, CastorCluster b) {
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
std::vector<std::vector<double> > KtAlgorithm::calcdPairs (CastorClusterCollection protoclusters, std::vector<std::vector<double> > dPairs) {
  int nRows = protoclusters.size();
  dPairs.reserve(nRows);
  // fill dPairs vector with distances
  for (int i=0;i<nRows-1;++i) {
  	std::vector<double> tempvec;
	tempvec.reserve(nRows);
  	for (int j=i+1;j<nRows;++j) {
		double D = calcDistanceDeltaR(protoclusters[i],protoclusters[j]);
		tempvec.push_back(D);	
	}
	dPairs.push_back(tempvec);
  }
  return dPairs;
}

// calculate ddi vectors
std::vector<double> KtAlgorithm::calcddi (CastorClusterCollection protoclusters, std::vector<double> ddi) {  
  int nRows = protoclusters.size();
  ddi.reserve(nRows);
  // fill ddi vector with energy squared
  for (int i=0;i<nRows;++i) {
  	ddi.push_back(protoclusters[i].energy()*protoclusters[i].energy());
  }
  return ddi;
}

// main public function executes Kt algorithm, is called to give results      
CastorClusterCollection KtAlgorithm::runKtAlgo (const CastorTowerRefVector& InputTowers, const int recom, const double rParameter) {
  if(debug) cout<<""<<endl;
  if(debug) cout<<"---------------------"<<endl;
  if(debug) cout<<"entering Kt algo code"<<endl;
  if(debug) cout<<"---------------------"<<endl;
  if(debug) cout<<""<<endl;

  // get and check input size
  int nTowers = InputTowers.size();
  if (nTowers==0) {
  	cout << "Warning: You are trying to run the KtAlgorithm with 0 input towers. \n";
  }
  
  // define output
  CastorClusterCollection protoclusters, clusters;
  protoclusters.reserve(InputTowers.size());

  // copy towers to protoclusters vector
  for (CastorTower_iterator it_tower = (InputTowers).begin(); it_tower != (InputTowers).end(); it_tower++) {
    reco::CastorTowerRef tower_p = *it_tower;
    TowerPoint temp(tower_p->rho(),tower_p->eta(),tower_p->phi());
    Point position(temp);
    CastorTowerRefVector usedTowers;
    usedTowers.push_back(tower_p);
    protoclusters.push_back(CastorCluster(tower_p->energy(),position,tower_p->emEnergy(),tower_p->hadEnergy(),tower_p->fem(),
					  0,tower_p->depth(),0,0,usedTowers));
  } 
  
  // start merging until only one cluster is left
  int ncluster;
  while (ncluster=protoclusters.size()>1) {
         
        // call calcddi and calcdPairs function
  	std::vector<double> ddi;
  	std::vector<std::vector<double> > dPairs;
  	ddi = calcddi(protoclusters,ddi);
	dPairs = calcdPairs(protoclusters,dPairs);
  
  	int iPairMin, jPairMin;
	double dPairMin, dClusterMin;
	
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
	int iClusterMin = 0;
	for (size_t i=0;i<ddi.size();i++) {
		if (ddi[i] < temp2) {
			temp2 = ddi[i];
			iClusterMin = i;
		}
	}
	dClusterMin = temp2;
	dClusterMin = dClusterMin*rParameter*rParameter;
	
	// take min dPairs and ddi and merge or not
	if ( dPairMin < dClusterMin ) {
		// merge the 2 protoclusters and put it in the protocluster list at iPairMin
		CastorCluster recombined = calcRecom(protoclusters[iPairMin],protoclusters[jPairMin],recom);
  		protoclusters[iPairMin] = recombined;
  		protoclusters.erase(protoclusters.begin()+jPairMin);
	} else {
		// put protocluster iClusterMin in list of final clusters and remove protocluster iClusterMin
		clusters.push_back(protoclusters[iClusterMin]);
		protoclusters.erase(protoclusters.begin()+iClusterMin);
	}
	
  }
  
  // end of loop, should have ncluster = 1
  clusters.push_back(protoclusters[0]); // if there's only one protocluster left, make a cluster of it, is this ok?
  
  // calculate width, fhot and sigma_z of all the clusters 
  double sum_e;
  double sum_distances;
  double clusterwidth;

  double clusterfhot;

  double weight;
  double zmean;
  double z2mean;
  double clustersigmaz;

  if(debug) cout<<endl;
  if(debug) cout<<"number of clusters in the event: "<<clusters.size()<<endl;

  // loop over clusters
  for (size_t i=0;i<clusters.size();i++) {
    CastorTowerRefVector usedTowers = clusters[i].getUsedTowers();

    if(debug) cout<<endl;
    if(debug) cout<<"cluster: "<<i+1<<" is made of: "<<usedTowers.size()<<" towers"<<endl;
    
    sum_e = 0.;
    sum_distances = 0.;
    clusterwidth = 0;
    
    clusterfhot = 0;
    
    weight = 0.;
    zmean = 0.;
    z2mean= 0.;
    clustersigmaz = 0;

    // loop over towers
    for (CastorTower_iterator it_tower = (usedTowers).begin(); it_tower != (usedTowers).end(); it_tower++) {
      reco::CastorTowerRef tower_p = *it_tower;

      sum_e+= tower_p->energy();
      sum_distances+= pow(phiangle(tower_p->phi() - clusters[i].phi()),2)*tower_p->energy();
      clusterfhot+=tower_p->fhot()*tower_p->energy(); 
      
      // loop over cells
      for (CastorCell_iterator it = tower_p->cellsBegin(); it != tower_p->cellsEnd(); it++) {
	reco::CastorCellRef cell_p = *it;

	Point rcell = cell_p->position();
	double Ecell = cell_p->energy();
	
	weight+=Ecell;
	zmean+=Ecell*cell_p->z();
	z2mean+=Ecell*cell_p->z()*cell_p->z();
      } // end loop over cells
    } // end loop over towers
    
    zmean/=weight;
    z2mean/=weight;
    double sigmaz2 = z2mean - zmean*zmean;
    if(sigmaz2 > 0) clustersigmaz = std::sqrt(sigmaz2);

    clusterwidth = sqrt(sum_distances/sum_e);
    clusterfhot/=sum_e;
    
    TowerPoint clustertemppos(clusters[i].rho(),clusters[i].eta(),clusters[i].phi());
    Point clusterpos(clustertemppos);
    clusters[i] = CastorCluster(clusters[i].energy(),clusterpos,clusters[i].emEnergy(),clusters[i].hadEnergy(),clusters[i].fem(),clusterwidth,
				clusters[i].depth(),clusterfhot,clustersigmaz,clusters[i].getUsedTowers()); 
    if(debug) cout<<endl;
    if(debug) cout<<"cluster: "<<i+1<<" sigma z: "<<clusters[i].sigmaz()<<" fhot: "<<clusters[i].fhot()<<" rho: "<<clusters[i].rho()<<endl;
    if(debug) getchar();
    if(debug) cout<<endl;
    if(debug) getchar();
  } // end loop over clusters
  
  return clusters;
}

