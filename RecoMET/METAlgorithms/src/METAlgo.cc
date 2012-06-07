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
void METAlgo::run(edm::Handle<edm::View<Candidate> > input, CommonMETData *met, double globalThreshold) 
{ 
  double sum_px = 0.0;
  double sum_py = 0.0;
  double sum_pz = 0.0;
  double sum_et = 0.0;
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
	double theta = candidate->theta();
	double e     = candidate->energy();
	double et    = e*sin(theta);
	sum_px += candidate->px();
	sum_py += candidate->py();
	sum_pz += candidate->pz();
	sum_et += et;
      }
  }
  met->mex   = -sum_px;
  met->mey   = -sum_py;
  met->mez   = -sum_pz;
  met->met   = sqrt( sum_px*sum_px + sum_py*sum_py );
  met->sumet = sum_et;
  met->phi   = atan2( -sum_py, -sum_px ); // since MET is now a candidate,
}                                         // this is no longer needed
//------------------------------------------------------------------------

