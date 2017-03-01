// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTElectronTrackIsolationProducers
// 
/**\class EgammaHLTElectronTrackIsolationProducers EgammaHLTElectronTrackIsolationProducers.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTElectronTrackIsolationProducers.h
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
//
// $Id: EgammaHLTElectronTrackIsolationProducers.h,v 1.3 2011/12/19 11:16:45 sani Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTElectronTrackIsolationProducers : public edm::global::EDProducer<> {
public:
  explicit EgammaHLTElectronTrackIsolationProducers(const edm::ParameterSet&);
  ~EgammaHLTElectronTrackIsolationProducers();
  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<reco::ElectronCollection> electronProducer_;
  const edm::EDGetTokenT<reco::TrackCollection> trackProducer_;
  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotProducer_;

  const bool useGsfTrack_;
  const bool useSCRefs_;

  const double egTrkIsoPtMin_; 
  const double egTrkIsoConeSize_;
  const double egTrkIsoZSpan_;   
  const double egTrkIsoRSpan_;  
  const double egTrkIsoVetoConeSizeBarrel_;
  const double egTrkIsoVetoConeSizeEndcap_;
  const double egTrkIsoStripBarrel_;
  const double egTrkIsoStripEndcap_;
};

