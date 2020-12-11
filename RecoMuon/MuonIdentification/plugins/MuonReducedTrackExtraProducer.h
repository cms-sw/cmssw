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

class MuonReducedTrackExtraProducer : public edm::stream::EDProducer<> {
public:
  MuonReducedTrackExtraProducer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<edm::View<reco::Muon>> muonToken_;
  std::vector<edm::EDGetTokenT<reco::TrackExtraCollection>> trackExtraTokens_;
  std::vector<edm::EDGetTokenT<edm::Association<reco::TrackExtraCollection>>> trackExtraAssocs_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> pixelClusterToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> stripClusterToken_;
  const bool outputClusters_;
  const StringCutObjectSelector<reco::Muon> selector_;
  const edm::EDPutTokenT<reco::TrackExtraCollection> trackExtraOutToken_;
  const edm::EDPutTokenT<TrackingRecHitCollection> trackingRecHitsOutToken_;
  edm::EDPutTokenT<edmNew::DetSetVector<SiPixelCluster>> pixelClusterOutToken_;
  edm::EDPutTokenT<edmNew::DetSetVector<SiStripCluster>> stripClusterOutToken_;
  const edm::EDPutTokenT<edm::Association<reco::TrackExtraCollection>> associationOutToken_;
};
#endif
