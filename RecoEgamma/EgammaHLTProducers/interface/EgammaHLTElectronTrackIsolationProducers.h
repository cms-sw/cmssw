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

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

class EgammaHLTElectronTrackIsolationProducers : public edm::EDProducer {
public:
  explicit EgammaHLTElectronTrackIsolationProducers(const edm::ParameterSet&);
  ~EgammaHLTElectronTrackIsolationProducers();
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:
  edm::EDGetTokenT<reco::ElectronCollection> electronProducer_;
  edm::EDGetTokenT<reco::TrackCollection> trackProducer_;
  edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotProducer_;

  bool useGsfTrack_;
  bool useSCRefs_;

  double egTrkIsoPtMin_; 
  double egTrkIsoConeSize_;
  double egTrkIsoZSpan_;   
  double egTrkIsoRSpan_;  
  double egTrkIsoVetoConeSizeBarrel_;
  double egTrkIsoVetoConeSizeEndcap_;
  double egTrkIsoStripBarrel_;
  double egTrkIsoStripEndcap_;
};

