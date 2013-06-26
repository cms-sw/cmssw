// File: METAlgo.cc
// Description:  see METAlgo.h
// Author: Michael Schmitt, Richard Cavanaugh The University of Florida
// Creation Date:  MHS May 31, 2005 Initial version.
//
//------------------------------------------------------------------------

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "RecoMET/METAlgorithms/interface/METAlgo.h"
#include <iostream>

using namespace std;
using namespace reco;

//------------------------------------------------------------------------
// Default Constructer
//----------------------------------
METAlgo::METAlgo() {}
//------------------------------------------------------------------------

//------------------------------------------------------------------------
// Default Destructor
//----------------------------------
METAlgo::~METAlgo() {}
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
//void METAlgo::run(const CandidateCollection *input, CommonMETData *met, double globalThreshold) 
void METAlgo::run(edm::Handle<edm::View<Candidate> > input, CommonMETData *met, double globalThreshold) 
{ 
  double sum_et = 0.0;
  double sum_ex = 0.0;
  double sum_ey = 0.0;
  double sum_ez = 0.0;
  // Loop over Candidate Objects and calculate MET and related quantities
  /*
  CandidateCollection::const_iterator candidate;
  for( candidate = input->begin(); candidate != input->end(); candidate++ )
  */
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
  met->mex   = -sum_ex;
  met->mey   = -sum_ey;
  met->mez   = -sum_ez;
  met->met   = sqrt( sum_ex*sum_ex + sum_ey*sum_ey );
  // cout << "MET = " << met->met << endl;
  met->sumet = sum_et;
  met->phi   = atan2( -sum_ey, -sum_ex ); // since MET is now a candidate,
}                                         // this is no longer needed
//------------------------------------------------------------------------

