// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      GenSpecificAlgo
// 
/**\class GenSpecificAlgo GenSpecificAlgo.h RecoMET/METAlgorithms/interface/GenSpecificAlgo.h

 Description: Adds generator level HEPMC specific information to MET

 Implementation:
     [Notes on implementation]
*/
//
// Original Authors:  R. Cavanaugh (taken from F.Ratnikov, UMd)
//          Created:  June 6, 2006
// $Id: GenSpecificAlgo.h,v 1.6 2013/05/03 18:52:47 salee Exp $
//
//
#ifndef METProducers_GenMETInfo_h
#define METProducers_GenMETInfo_h

//____________________________________________________________________________||
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/METReco/interface/SpecificGenMETData.h"

//____________________________________________________________________________||
class GenSpecificAlgo
{

public:
  reco::GenMET addInfo(edm::Handle<edm::View<reco::Candidate> > particles, CommonMETData *met, double globalThreshold = 0, bool onlyFiducial = false,bool applyFiducialThresholdForFractions = false, bool usePt = false);

private:
  typedef math::XYZTLorentzVector LorentzVector;
  typedef math::XYZPoint Point;

  void fillCommonMETData(CommonMETData *met, edm::Handle<edm::View<reco::Candidate> >& particles, double globalThreshold, bool onlyFiducial, bool usePt);
  SpecificGenMETData mkSpecificGenMETData(edm::Handle<edm::View<reco::Candidate> >& particles,double globalThreshold, bool onlyFiducial,bool applyFiducialThresholdForFractions, bool usePt);

};

//____________________________________________________________________________||
#endif // METProducers_GenMETInfo_h
