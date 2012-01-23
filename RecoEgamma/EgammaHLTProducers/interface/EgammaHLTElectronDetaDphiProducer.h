// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTElectronDetaDphiProducer
// 
/**\class EgammaHLTElectronDetaDphiProducer EgammaHLTElectronDetaDphiProducer.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTElectronDetaDphiProducer.h
*/
//
// Original Author:  Roberto Covarelli (CERN)
//
// $Id: EgammaHLTElectronDetaDphiProducer.h,v 1.2 2011/12/19 11:16:45 sani Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

class MagneticField;

#include "TTree.h"
//
// class declaration
//

class EgammaHLTElectronDetaDphiProducer : public edm::EDProducer {
   public:
      explicit EgammaHLTElectronDetaDphiProducer(const edm::ParameterSet&);
      ~EgammaHLTElectronDetaDphiProducer();


      virtual void produce(edm::Event&, const edm::EventSetup&);
  
   private:
  std::pair<float,float> calDEtaDPhiSCTrk(reco::ElectronRef& eleref, const reco::BeamSpot::Point& BSPosition,const MagneticField *magField);
  static reco::ElectronRef getEleRef(const reco::RecoEcalCandidateRef& recoEcalCandRef,const edm::Handle<reco::ElectronCollection>& electronHandle);
  
  edm::InputTag electronProducer_;
  edm::InputTag recoEcalCandidateProducer_;

  bool useSCRefs_;
  bool useTrackProjectionToEcal_;
  edm::InputTag bsProducer_;
  
};

