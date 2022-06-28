#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "DataFormats/L1TParticleFlow/interface/PFTrack.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TPFUtils.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/ParametricResolution.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

namespace l1tpf {
  class PFTrackProducerFromL1Tracks : public edm::stream::EDProducer<> {
  public:
    explicit PFTrackProducerFromL1Tracks(const edm::ParameterSet &);
    ~PFTrackProducerFromL1Tracks() override {}

  private:
    edm::EDGetTokenT<std::vector<l1t::PFTrack::L1TTTrackType>> TrackTag_;
    edm::ESWatcher<IdealMagneticFieldRecord> BFieldWatcher_;
    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> BFieldTag_;
    int nParam_;
    float fBz_;
    l1tpf::ParametricResolution resolCalo_, resolTrk_;
    std::vector<StringCutObjectSelector<l1t::PFTrack::L1TTTrackType>> qualityBitSetters_;
    bool redigitizeTrackWord_;

    void produce(edm::Event &, const edm::EventSetup &) override;

  };  // class
}  // namespace l1tpf

l1tpf::PFTrackProducerFromL1Tracks::PFTrackProducerFromL1Tracks(const edm::ParameterSet &iConfig)
    : TrackTag_(consumes<std::vector<l1t::PFTrack::L1TTTrackType>>(iConfig.getParameter<edm::InputTag>("L1TrackTag"))),
      BFieldTag_{esConsumes<MagneticField, IdealMagneticFieldRecord>()},
      nParam_(iConfig.getParameter<unsigned int>("nParam")),
      resolCalo_(iConfig.getParameter<edm::ParameterSet>("resolCalo")),
      resolTrk_(iConfig.getParameter<edm::ParameterSet>("resolTrack")),
      redigitizeTrackWord_(iConfig.getParameter<bool>("redigitizeTrackWord")) {
  for (const auto &cut : iConfig.getParameter<std::vector<std::string>>("qualityBits")) {
    qualityBitSetters_.emplace_back(cut);
  }
  produces<l1t::PFTrackCollection>();
}

void l1tpf::PFTrackProducerFromL1Tracks::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  if (BFieldWatcher_.check(iSetup)) {
    fBz_ = iSetup.getData(BFieldTag_).inTesla(GlobalPoint(0, 0, 0)).z();
  }

  std::unique_ptr<l1t::PFTrackCollection> out(new l1t::PFTrackCollection());

  // https://github.com/skinnari/cmssw/blob/80c19f1b721325c3a02ee0482f72fb974a4c3bf7/L1Trigger/TrackFindingTracklet/test/L1TrackNtupleMaker.cc
  edm::Handle<std::vector<l1t::PFTrack::L1TTTrackType>> htracks;
  iEvent.getByToken(TrackTag_, htracks);
  const auto &tracks = *htracks;

  for (unsigned int i = 0, n = tracks.size(); i < n; ++i) {
    const auto &tk = tracks[i];

    float pt = tk.momentum().perp();
    float eta = tk.momentum().eta();
    float phi = tk.momentum().phi();
    float z0 = tk.POCA().z();  //cm
    int charge = tk.rInv() > 0 ? +1 : -1;

    reco::Candidate::PolarLorentzVector p4p(pt, eta, phi, 0.137);  // pion mass
    reco::Particle::LorentzVector p4(p4p.X(), p4p.Y(), p4p.Z(), p4p.E());
    reco::Particle::Point vtx(tk.POCA().x(), tk.POCA().y(), z0);

    auto caloetaphi = l1tpf::propagateToCalo(p4, math::XYZTLorentzVector(0., 0., z0, 0.), charge, fBz_);

    float trkErr = resolTrk_(pt, std::abs(eta));
    float caloErr = resolCalo_(pt, std::abs(eta));
    int quality = 0;
    for (int i = 0, n = qualityBitSetters_.size(), bit = 1; i < n; ++i, bit <<= 1) {
      if (qualityBitSetters_[i](tk))
        quality += bit;
    }
    out->emplace_back(charge,
                      p4,
                      vtx,
                      l1t::PFTrack::TrackRef(htracks, i),
                      nParam_,
                      caloetaphi.first,
                      caloetaphi.second,
                      trkErr,
                      caloErr,
                      quality);

    if (redigitizeTrackWord_) {
      // Force re-digitization if an old TTrack object is read from an EDM file, and update the quaility bit for now
      l1t::PFTrack::L1TTTrackType trackCopy = tk;
      trackCopy.setTrackWordBits();  // important
      TTTrack_TrackWord &tw = out->back().trackWord();
      tw.setTrackWord(trackCopy.getValidWord(),
                      trackCopy.getRinvWord(),
                      trackCopy.getPhiWord(),
                      trackCopy.getTanlWord(),
                      trackCopy.getZ0Word(),
                      trackCopy.getD0Word(),
                      trackCopy.getChi2RPhiWord(),
                      trackCopy.getChi2RZWord(),
                      trackCopy.getBendChi2Word(),
                      trackCopy.getHitPatternWord(),
                      trackCopy.getMVAQualityWord(),
                      ap_uint<TTTrack_TrackWord::kMVAOtherSize>(quality));
    }
  }
  iEvent.put(std::move(out));
}
using l1tpf::PFTrackProducerFromL1Tracks;
DEFINE_FWK_MODULE(PFTrackProducerFromL1Tracks);
