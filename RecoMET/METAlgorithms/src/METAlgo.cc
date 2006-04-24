// File: METAlgo.cc
// Description:  see METAlgo.h
// Author: Michael Schmitt, Richard Cavanaugh The University of Florida
// Creation Date:  MHS May 31, 2005 Initial version.
//
//--------------------------------------------

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "RecoMET/METAlgorithms/interface/METAlgo.h"
#include <iostream>

using namespace std;
using namespace reco;

METAlgo::METAlgo() {}

METAlgo::~METAlgo() {}

void METAlgo::run(const InputCollection &input, METCollection &metvec) 
{
  // Clean up the EDProduct, it should be empty
  metvec.clear();
  double sum_et = 0.0;
  double sum_ex = 0.0;
  double sum_ey = 0.0;
  double sum_ez = 0.0;
  // Loop over Candidate Objects and calculate MET quantities
  METAlgo::InputCollection::const_iterator candidate;
  for( candidate = input.begin(); candidate != input.end(); candidate++ )
    {
      double phi   = (*candidate)->phi();
      double theta = (*candidate)->theta();
      double e     = (*candidate)->energy();
      double et    = e*sin(theta);
      sum_ez += e*cos(theta);
      sum_et += et;
      sum_ex += et*cos(phi);
      sum_ey += et*sin(phi);
    }
  // Create a holder for the "met" data
  CommonMETData met;
  met.mex   = -sum_ex;
  met.mey   = -sum_ey;
  met.mez   = -sum_ez;
  met.met   = sqrt( sum_ex*sum_ex + sum_ey*sum_ey );
  met.sumet = sum_et;
  met.phi   = atan2( -sum_ey, -sum_ex );
  // Save result: create MET object initialised with "met" data
  MET result( met );
  metvec.push_back(result);
}

