// File: TCMETAlgo.cc
// Description:  see TCMETAlgo.h
// Author: F. Golf, A. Yagil
// Creation Date:  Nov 12, 2008 Initial version.
//
//------------------------------------------------------------------------

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "RecoMET/METAlgorithms/interface/TCMETAlgo.h"
#include <iostream>

using namespace std;
using namespace reco;

//------------------------------------------------------------------------
// Default Constructer
//----------------------------------
TCMETAlgo::TCMETAlgo() {}
//------------------------------------------------------------------------

//------------------------------------------------------------------------
// Default Destructor
//----------------------------------
TCMETAlgo::~TCMETAlgo() {}
//------------------------------------------------------------------------

//------------------------------------------------------------------------
// This method represents "the" implementation of the MET algorithm and is
// very simple:
// (1) It takes as input a collection of candidates (which can be
// calorimeter towers, HEPMC generator-level particles, etc).
// (2) It returns as output, a pointer to a struct of CommonMETData which
// contains the following members:  MET, MEx, MEy, SumET, and MEz
// (The inclusion of MEz deserves some justification ; it is included here
// since it _may_ be useful for Data Quality Monitering as it should be 
// symmetrically distributed about the origin.)
//----------------------------------

void TCMETAlgo::run(edm::Handle<edm::View<Candidate> > input, CommonMETData *tcmet, double globalThreshold) 
{ 
  double sum_et = 0.0;
  double sum_ex = 0.0;
  double sum_ey = 0.0;
  double sum_ez = 0.0;
  // Loop over Candidate Objects and calculate TCMET and related quantities

  for (unsigned int candidate_i = 0; candidate_i < input->size(); candidate_i++)
  {
    const Candidate *candidate = &((*input)[candidate_i]);
    if( candidate->et() > globalThreshold  ) 
      {
	double phi   = candidate->phi();
	double theta = candidate->theta();
	double e     = candidate->energy();
	double et    = e*sin(theta);
	sum_ez += e*cos(theta);
	sum_et += et;
	sum_ex += et*cos(phi);
	sum_ey += et*sin(phi);
      }
  }
  tcmet->mex   = -sum_ex;
  tcmet->mey   = -sum_ey;
  tcmet->mez   = -sum_ez;
  tcmet->met   = sqrt( sum_ex*sum_ex + sum_ey*sum_ey );
  tcmet->sumet = sum_et;
  tcmet->phi   = atan2( -sum_ey, -sum_ex ); 
//------------------------------------------------------------------------

