#ifndef TauJetSelectorForHLTTrackSeeding_H
#define TauJetSelectorForHLTTrackSeeding_H

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/TrackJet.h"
#include "DataFormats/JetReco/interface/TrackJetCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class TauJetSelectorForHLTTrackSeeding : public edm::global::EDProducer<> {

public:
  explicit TauJetSelectorForHLTTrackSeeding(const edm::ParameterSet&);
  ~TauJetSelectorForHLTTrackSeeding();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() ;
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  virtual void endJob() ;
      
  // ----------member data ---------------------------

  edm::EDGetTokenT<reco::TrackJetCollection> inputTrackJetToken_;
  edm::EDGetTokenT<reco::CaloJetCollection> inputCaloJetToken_;
  edm::EDGetTokenT<reco::TrackCollection> inputTrackToken_;
  const double ptMinCaloJet_;
  const double etaMinCaloJet_;
  const double etaMaxCaloJet_;
  const double tauConeSize_;
  const double isolationConeSize_;
  const double fractionMinCaloInTauCone_;
  const double fractionMaxChargedPUInCaloCone_;
  const double ptTrkMaxInCaloCone_;
  const int nTrkMaxInCaloCone_;
};
#endif

