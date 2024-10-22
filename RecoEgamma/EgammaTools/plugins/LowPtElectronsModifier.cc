#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoEgamma/EgammaTools/interface/LowPtConversion.h"

////////////////////////////////////////////////////////////////////////////////
//
class LowPtElectronModifier : public ModifyObjectValueBase {
public:
  LowPtElectronModifier(const edm::ParameterSet& conf, edm::ConsumesCollector&);
  ~LowPtElectronModifier() override = default;

  void setEvent(const edm::Event&) final;
  void setEventContent(const edm::EventSetup&) final;

  void modifyObject(pat::Electron& ele) const final;

private:
  const edm::EDGetTokenT<reco::ConversionCollection> convT_;
  reco::ConversionCollection const* conv_ = nullptr;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotT_;
  reco::BeamSpot const* beamSpot_ = nullptr;
  const edm::EDGetTokenT<reco::VertexCollection> verticesT_;
  reco::VertexCollection const* vertices_ = nullptr;
  bool extra_;
};

////////////////////////////////////////////////////////////////////////////////
//
LowPtElectronModifier::LowPtElectronModifier(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
    : ModifyObjectValueBase(conf),
      convT_(cc.consumes<reco::ConversionCollection>(conf.getParameter<edm::InputTag>("conversions"))),
      conv_(),
      beamSpotT_(cc.consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpot"))),
      beamSpot_(),
      verticesT_(cc.consumes<reco::VertexCollection>(conf.getParameter<edm::InputTag>("vertices"))),
      vertices_(),
      extra_(conf.getParameter<bool>("addExtraUserVars")) {
  ;
}

////////////////////////////////////////////////////////////////////////////////
//
void LowPtElectronModifier::setEvent(const edm::Event& iEvent) {
  conv_ = &iEvent.get(convT_);
  beamSpot_ = &iEvent.get(beamSpotT_);
  vertices_ = &iEvent.get(verticesT_);
}

////////////////////////////////////////////////////////////////////////////////
//
void LowPtElectronModifier::setEventContent(const edm::EventSetup& iSetup) {}

////////////////////////////////////////////////////////////////////////////////
//
void LowPtElectronModifier::modifyObject(pat::Electron& ele) const {
  // Embed Conversion info
  LowPtConversion conv;
  conv.match(*beamSpot_, *conv_, ele);
  conv.addUserVars(ele);
  if (extra_) {
    conv.addExtraUserVars(ele);
  }
  // Set impact parameters
  auto const& gsfTrack = *ele.gsfTrack();
  if (!vertices_->empty()) {
    const reco::Vertex& pv = vertices_->front();
    ele.setDB(gsfTrack.dxy(pv.position()),
              gsfTrack.dxyError(pv.position(), pv.covariance()),
              pat::Electron::PV2D);  // PV2D
    ele.setDB(gsfTrack.dz(pv.position()), std::hypot(gsfTrack.dzError(), pv.zError()),
              pat::Electron::PVDZ);  // PVDZ
  }
  ele.setDB(gsfTrack.dxy(*beamSpot_), gsfTrack.dxyError(*beamSpot_),
            pat::Electron::BS2D);  // BS2D
}

////////////////////////////////////////////////////////////////////////////////
//
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory, LowPtElectronModifier, "LowPtElectronModifier");
