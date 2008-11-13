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
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include <cmath>
#include <iostream>
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"




using namespace std;
using namespace reco;
using namespace math;
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

reco::MET TCMETAlgo::addInfo(edm::Handle<edm::View<Candidate> > input, CommonMETData *TCMETData, bool NoHF, double globalThreshold) 
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
  TCMETData->mex   = -sum_ex;
  TCMETData->mey   = -sum_ey;
  TCMETData->mez   = -sum_ez;
  TCMETData->met   = sqrt( sum_ex*sum_ex + sum_ey*sum_ey );
  TCMETData->sumet = sum_et;
  TCMETData->phi   = atan2( -sum_ey, -sum_ex ); 

  XYZTLorentzVector p4( TCMETData->mex , TCMETData->mey , 0, TCMETData->met);
  XYZPointD vtx(0,0,0);
  MET tcmet(TCMETData->sumet, p4, vtx);
  return tcmet;
//------------------------------------------------------------------------
}
