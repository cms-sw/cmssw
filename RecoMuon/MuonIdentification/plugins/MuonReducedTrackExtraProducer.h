#ifndef RecoMuon_MuonIdentification_MuonReducedTrackExtraProducer_H
#define RecoMuon_MuonIdentification_MuonReducedTrackExtraProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/Common/interface/Association.h"

class SiPixelCluster;
class SiStripCluster;

namespace pat {
  class Muon;
}

class MuonReducedTrackExtraProducer : public edm::stream::EDProducer<> {
public:
  MuonReducedTrackExtraProducer(const edm::ParameterSet&);

  ~MuonReducedTrackExtraProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<edm::View<reco::Muon>> muonToken_;
  std::vector<edm::EDGetTokenT<reco::TrackExtraCollection>> trackExtraTokens_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> pixelClusterToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> stripClusterToken_;
  std::string cut_;
  bool outputClusters_;
  StringCutObjectSelector<pat::Muon> selector_;
  edm::EDPutTokenT<reco::TrackExtraCollection> trackExtraOutToken_;
  edm::EDPutTokenT<TrackingRecHitCollection> trackingRecHitsOutToken_;
  edm::EDPutTokenT<edmNew::DetSetVector<SiPixelCluster>> pixelClusterOutToken_;
  edm::EDPutTokenT<edmNew::DetSetVector<SiStripCluster>> stripClusterOutToken_;
  edm::EDPutTokenT<edm::Association<reco::TrackExtraCollection>> associationOutToken_;
};
#endif
