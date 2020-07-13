#ifndef MuonIdentification_MuonTrackExtraThinningProducer_h
#define MuonIdentification_MuonTrackExtraThinningProducer_h

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/ThinningProducer.h"
#include "FWCore/Framework/interface/stream/ThinningSelectorByRefBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

namespace edm {
  class ParameterSetDescription;
}

class MuonTrackExtraSelector : public edm::ThinningSelectorByRefBase<reco::TrackExtraCollection> {
public:
  MuonTrackExtraSelector(edm::ParameterSet const& pset, edm::ConsumesCollector&& cc);
  static void fillDescription(edm::ParameterSetDescription& desc);
  void preChooseRefs(edm::Handle<reco::TrackExtraCollection> trackExtras,
                     edm::Event const& event,
                     edm::EventSetup const& es) override;
  void modify(reco::TrackExtra& trackExtra) override;

private:
  std::string cut_;
  bool slimTrajParams_;
  bool slimResiduals_;
  bool slimFinalState_;
  edm::EDGetTokenT<edm::View<reco::Muon> > muonToken_;
  StringCutObjectSelector<reco::Muon> selector_;
};

typedef edm::ThinningProducer<reco::TrackExtraCollection, MuonTrackExtraSelector> MuonTrackExtraThinningProducer;

#endif
