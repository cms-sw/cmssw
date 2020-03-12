#ifndef HLTMuonTrackSelector_h
#define HLTMuonTrackSelector_h

/*
* class HLTMuonTrackSelector
* 
* Select tracks matched to the reco::Muon
* 
* base on RecoTracker/FinalTrackSelectors/plugins/TrackCollectionFilterCloner.cc
* 
* Author: Kyeongpil Lee (kplee@cern.ch)
* 
*/

#include "RecoTracker/FinalTrackSelectors/src/TrackCollectionCloner.cc"

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include <vector>

class HLTMuonTrackSelector : public edm::global::EDProducer<> {
public:
  explicit HLTMuonTrackSelector(const edm::ParameterSet &);
  ~HLTMuonTrackSelector() override;

  using MVACollection = std::vector<float>;

  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  TrackCollectionCloner collectionCloner;
  const TrackCollectionCloner::Tokens collectionClonerTokens;

  const edm::EDGetTokenT<std::vector<reco::Muon> > token_muon;
  const edm::EDGetTokenT<MVACollection> token_originalMVAVals;
  const bool flag_copyMVA;
};

#endif  //HLTMuonTrackSelector_h
