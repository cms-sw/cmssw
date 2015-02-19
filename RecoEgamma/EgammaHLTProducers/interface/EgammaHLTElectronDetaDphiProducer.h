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
// $Id: EgammaHLTElectronDetaDphiProducer.h,v 1.5 2012/02/10 22:41:25 dmytro Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

class MagneticField;

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTElectronDetaDphiProducer : public edm::stream::EDProducer<> {
public:
  explicit EgammaHLTElectronDetaDphiProducer(const edm::ParameterSet&);
  ~EgammaHLTElectronDetaDphiProducer();
  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  std::pair<float,float> calDEtaDPhiSCTrk(reco::ElectronRef& eleref, const reco::BeamSpot::Point& BSPosition,const MagneticField *magField);
  static reco::ElectronRef getEleRef(const reco::RecoEcalCandidateRef& recoEcalCandRef,const edm::Handle<reco::ElectronCollection>& electronHandle);
  
  const edm::EDGetTokenT<reco::ElectronCollection> electronProducer_;
  const edm::EDGetTokenT<reco::BeamSpot> bsProducer_;
  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  
  const bool useSCRefs_;
  const bool useTrackProjectionToEcal_;
  const bool variablesAtVtx_;

  const MagneticField* magField_;
};

