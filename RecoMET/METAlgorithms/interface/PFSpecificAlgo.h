// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      PFSpecificAlgo
// 
/**\class PFSpecificAlgo PFSpecificAlgo.h RecoMET/METAlgorithms/interface/PFSpecificAlgo.h

 Description: Adds Particle Flow specific information to MET

 Implementation:
     [Notes on implementation]
*/
//
// Original Authors:  R. Remington (UF), R. Cavanaugh (UIC/Fermilab)
//          Created:  October 27, 2008
// $Id: PFSpecificAlgo.h,v 1.6 2012/06/10 16:37:16 sakuma Exp $
//
//
#ifndef METAlgorithms_PFMETInfo_h
#define METAlgorithms_PFMETInfo_h

//____________________________________________________________________________||
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/SpecificPFMETData.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "RecoMET/METAlgorithms/interface/SignPFSpecificAlgo.h"
#include "TMatrixD.h"

namespace metsig {
  class SignAlgoResolutions;
}

//____________________________________________________________________________||
class PFSpecificAlgo
{
 public:
  PFSpecificAlgo() : doSignificance(false) { }
  
  void runSignificance(metsig::SignAlgoResolutions & resolutions, edm::Handle<edm::View<reco::PFJet> > jets);
  reco::PFMET addInfo(edm::Handle<edm::View<reco::Candidate> > PFCandidates, CommonMETData met);

 private:
  typedef math::XYZTLorentzVector LorentzVector;
  typedef math::XYZPoint Point;
  void initializeSpecificPFMETData(SpecificPFMETData &specific);
  SpecificPFMETData mkSpecificPFMETData(edm::Handle<edm::View<reco::Candidate> > &PFCandidates);
  TMatrixD mkSignifMatrix(edm::Handle<edm::View<reco::Candidate> > &PFCandidates);

  bool doSignificance;
  metsig::SignPFSpecificAlgo pfsignalgo_;
};

//____________________________________________________________________________||
#endif // METAlgorithms_PFMETInfo_h

