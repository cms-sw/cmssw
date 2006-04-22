// File: METAlgo.cc
// Description:  see METAlgo.h
// Author: Michael Schmitt, Richard Cavanaugh The University of Florida
// Creation Date:  MHS May 31, 2005 Initial version.
//
//--------------------------------------------

#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoMET/METAlgorithms/interface/METAlgo.h"
#include <iostream>

using namespace std;

METAlgo::METAlgo() {}

METAlgo::~METAlgo() {}

void METAlgo::run(const InputCollection &input, TowerMETCollection &metvec) 
{
  // Clean up the EDProduct, it should be empty
  metvec.clear();
  double sum_et = 0.;
  double sum_ex = 0.;
  double sum_ey = 0.;
  double sum_ez = 0.;
  // Loop over CaloTowers
  METAlgo::InputCollection::const_iterator tower;
  for( tower = input.begin(); tower != input.end(); tower++ )
    {
      double phi   = (*tower)->phi();
      double theta = (*tower)->theta();
      double e     = (*tower)->energy();
      double et    = e*sin(theta);
      sum_ez += e*cos(theta);
      sum_et += et;
      sum_ex += et*cos(phi);
      sum_ey += et*sin(phi);
    }
  // Calculate the Resultant MET Angle
  double met_phi = atan2( -sum_ey, -sum_ex ); 
  // Create a holder for the MET object
  TowerMET met;
  met.clearMET();
  // Set new TowerMET values
  //met.setLabel("");
  met.setMET( sqrt( sum_ex*sum_ex + sum_ey*sum_ey ) );
  met.setMEx(  -sum_ex  );
  met.setMEy(  -sum_ey  );
  met.setMEz(  -sum_ez  );
  met.setPhi(   met_phi );
  met.setSumET( sum_et  );
  // Raw MET is pushed into the zero position
  metvec.push_back(met);

}

