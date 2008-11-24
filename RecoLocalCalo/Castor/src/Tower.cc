// -*- C++ -*-
//
// Package:    Castor
// Class:      Castor
// 
/**\class Castor Tower.cc RecoLocalCalo/Castor/src/Tower.cc

 Description: Functions to make CastorTowers from CastorCells

 Implementation:
     
*/
//
// Original Author:  Hans Van Haevermaet
//         Created:  Sat May 24 12:00:56 CET 2008
// $Id: Tower.cc,v 1.1.2.2 2008/09/05 14:16:32 hvanhaev Exp $
//
//

// includes
#include <iostream>
#include <algorithm>
#include "DataFormats/CastorReco/interface/CastorCell.h"
#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorJet.h"
#include "DataFormats/CastorReco/interface/CastorEgamma.h"
#include "RecoLocalCalo/Castor/interface/Tower.h"

// namespaces
using namespace std;
using namespace reco;
using namespace math;

// typedefs
typedef math::XYZPointD Point;
typedef ROOT::Math::RhoZPhiPoint CellPoint;
typedef ROOT::Math::RhoEtaPhiPoint TowerPoint;

// main public function being executed, is called to give results      
CastorTowerCollection Tower::runTowerProduction (const CastorCellCollection inputcells, const double eta) {
  
  // get and check input size
  int nCells = inputcells.size();
  if (nCells==0) {
  	cout << "Warning: You are trying to run the Tower algorithm with 0 input cells. \n";
  }
  
  // define output
  CastorTowerCollection towers;
  int size = 16;
  towers.reserve(size);
  
  CastorCellCollection cells = inputcells;
  
  // declare castor array
  // (0,x): Energies - (1,x): emEnergies - (2,x): hadEnergies - (3,x): phi position
  double castortowerarray [4][16]; 
  std::vector<CastorCell> castorusedcellsarray [16];
  // set phi values of array sectors and everything else to zero
  for (int j = 0; j < 16; j++) {
	castortowerarray[3][j] = -2.94524 + j*0.3927;
	castortowerarray[0][j] = 0.;
	castortowerarray[1][j] = 0.;
	castortowerarray[2][j] = 0.;
  }
  
  // sort all cells into the arrays
  for (size_t i=0;i<cells.size();i++) {
  	for ( int j=0;j<16;j++) {
		if (cells[i].phi() == castortowerarray[3][j]) {
			if (fabs(cells[i].z()) < 14488) {
				castortowerarray[0][j] = castortowerarray[0][j] + cells[i].energy();
				castortowerarray[1][j] = castortowerarray[1][j] + cells[i].energy();
			} else {
				castortowerarray[0][j] = castortowerarray[0][j] + cells[i].energy();
				castortowerarray[2][j] = castortowerarray[2][j] + cells[i].energy();
			}
			castorusedcellsarray[j].push_back(cells[i]);
		}
	}
	
  }
  
  // make towers of the arrays
  for (int k=0;k<16;k++) {
  	if (castortowerarray[0][k] > 0.) {
  	TowerPoint temptowerposition(1.0,eta,castortowerarray[3][k]);
	Point towerposition(temptowerposition);
	double emtotRatio;
	emtotRatio = castortowerarray[1][k]/castortowerarray[0][k];
	double depth; 
	double sum = 0.; 
	double totalenergy = 0.;
	CastorCellCollection usedCells = castorusedcellsarray[k];
	for (size_t i=0;i<usedCells.size();i++) {
		sum = sum + usedCells[i].energy()*usedCells[i].z();
		totalenergy = totalenergy + usedCells[i].energy();
	}
	depth = sum/totalenergy;
	//AlgoId algo = fastsim;
	CastorTower newtower(castortowerarray[0][k],towerposition,castortowerarray[1][k],castortowerarray[2][k],emtotRatio,0.3927,depth,castorusedcellsarray[k]);
	towers.push_back(newtower);
	}
  }
  
  
  return towers;
}

