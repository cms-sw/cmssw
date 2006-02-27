// File: TowerMETAlgo.cc
// Description:  see TowerMETAlgo.h
// Author: Michael Schmitt, Richard Cavanaugh The University of Florida
// Creation Date:  MHS May 31, 2005 Initial version.
//
//--------------------------------------------

#include "RecoMET/METAlgorithms/interface/TowerMETAlgo.h"
#include <iostream>

using namespace std;

TowerMETAlgo::TowerMETAlgo() {}

TowerMETAlgo::~TowerMETAlgo() {}

void TowerMETAlgo::run(const CaloTowerCollection *towers, TowerMETCollection &metvec) 
{
  // Clean up the EDProduct, it should be empty
  metvec.clear();
  double sum_et = 0.;
  double sum_ex = 0.;
  double sum_ey = 0.;
  double sum_ez = 0.;
  // Loop over CaloTowers
  CaloTowerCollection::const_iterator ct_iter;
  for (ct_iter = towers->begin(); ct_iter != towers->end(); ct_iter++) 
    {
      // Get the relevant CaloTower data
      //double e     = ct_iter->e;
      //double theta = ct_iter->getTheta();
      double phi   = ct_iter->phi;
      // Sum over the transverse energy
      double et = ct_iter->eT;//10.0 * e * sin(theta);
      double eta = ct_iter->eta;
      double theta = 2.0 * atan( exp( -eta ) );
      double e = et * sin(theta);
      sum_ez += e * cos(theta);
      sum_et += et;
      sum_ex += et * cos(phi);
      sum_ey += et * sin(phi);
    }
  // Calculate the Resultant MET Angle
  double met_phi = atan2( -sum_ex, -sum_ey );
  // Create a holder for the MET object
  TowerMET met;
  met.clearMET();
  // Set new TowerMET values
  met.setLabel("");
  met.setMET(sqrt(sum_ex * sum_ex + sum_ey * sum_ey));
  met.setMEx(-sum_ex);
  met.setMEy(-sum_ey);
  met.setMEz(-sum_ez);
  met.setPhi(met_phi);
  met.setSumET(sum_et);
  // Raw MET is pushed into the zero position
  metvec.push_back(met);
}

