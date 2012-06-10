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
// $Id: METAlgo.h,v 1.12 2012/06/08 00:51:27 sakuma Exp $
//
//
#ifndef METAlgorithms_PFMETInfo_h
#define METAlgorithms_PFMETInfo_h

//____________________________________________________________________________||
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "RecoMET/METAlgorithms/interface/SignPFSpecificAlgo.h"
#include "DataFormats/JetReco/interface/PFJet.h"


class PFSpecificAlgo
{
 public:
  PFSpecificAlgo() : alsocalcsig(false), pfsignalgo_() {;}
  
  typedef math::XYZTLorentzVector LorentzVector;
  typedef math::XYZPoint Point;
  void runSignificance(metsig::SignAlgoResolutions & resolutions, edm::Handle<edm::View<reco::PFJet> > jets);
  reco::PFMET addInfo(edm::Handle<edm::View<reco::Candidate> > PFCandidates, CommonMETData met);

 private:
  bool alsocalcsig;
  metsig::SignPFSpecificAlgo pfsignalgo_;
};

//____________________________________________________________________________||
#endif // METAlgorithms_PFMETInfo_h

