// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      SignCaloSpecificAlgo
// 
/**\class SignCaloSpecificAlgo SignCaloSpecificAlgo.h RecoMET/METAlgorithms/interface/SignCaloSpecificAlgo.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Authors:  Kyle Story, Freya Blekman (Cornell University)
//          Created:  Fri Apr 18 11:58:33 CEST 2008
//
// 
#ifndef METProducers_SignCaloMETAlgo_h
#define METProducers_SignCaloMETAlgo_h

//____________________________________________________________________________||
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/SigInputObj.h"

namespace metsig {
  class SignAlgoResolutions;
}

//____________________________________________________________________________||
class SignCaloSpecificAlgo {

public:

  SignCaloSpecificAlgo();
  ~SignCaloSpecificAlgo();

  void usePreviousSignif(const std::vector<double> &values);
  void usePreviousSignif(const reco::METCovMatrix &matrix) { matrix_ = matrix; }
  double getSignificance(){return significance_;}
  reco::METCovMatrix getSignificanceMatrix()const {return matrix_;}

  void calculateBaseCaloMET(edm::Handle<edm::View<reco::Candidate> > towers,  const CommonMETData& met, const metsig::SignAlgoResolutions & resolutions, bool noHF, double globalthreshold);
  
 private:
  
  std::vector<metsig::SigInputObj> makeVectorOutOfCaloTowers(edm::Handle<edm::View<reco::Candidate> > towers, const metsig::SignAlgoResolutions& resolutions, bool noHF, double globalthreshold);
  
  double significance_;
  reco::METCovMatrix matrix_;
};


//____________________________________________________________________________||
#endif // METProducers_SignCaloMETAlgo_h
