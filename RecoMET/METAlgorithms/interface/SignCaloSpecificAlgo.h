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
// $Id: SignCaloSpecificAlgo.h,v 1.7 2012/09/17 17:54:30 veelken Exp $
//
// 
#ifndef METProducers_SignCaloMETAlgo_h
#define METProducers_SignCaloMETAlgo_h

//____________________________________________________________________________||
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/SigInputObj.h"
#include "TMatrixD.h"

namespace metsig {
  class SignAlgoResolutions;
}

//____________________________________________________________________________||
class SignCaloSpecificAlgo {

public:

  SignCaloSpecificAlgo();
  ~SignCaloSpecificAlgo();

  void usePreviousSignif(const std::vector<double> &values);
  void usePreviousSignif(const TMatrixD &matrix) { matrix_ = matrix; }
  double getSignificance(){return significance_;}
  TMatrixD getSignificanceMatrix()const {return matrix_;}

  void calculateBaseCaloMET(edm::Handle<edm::View<reco::Candidate> > towers,  CommonMETData met, const metsig::SignAlgoResolutions & resolutions, bool noHF, double globalthreshold);
  
 private:
  
  std::vector<metsig::SigInputObj> makeVectorOutOfCaloTowers(edm::Handle<edm::View<reco::Candidate> > towers, const metsig::SignAlgoResolutions& resolutions, bool noHF, double globalthreshold);
  
  double significance_;
  TMatrixD matrix_;
};


//____________________________________________________________________________||
#endif // METProducers_SignCaloMETAlgo_h
