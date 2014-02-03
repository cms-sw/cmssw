#include <boost/foreach.hpp>
#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"
#include "TrackingTools/IPTools/interface/IPTools.h"

class PFRecoTauDiscriminationByFlight : public PFTauDiscriminationProducerBase {
  public:
    PFRecoTauDiscriminationByFlight(const edm::ParameterSet& pset);
    virtual ~PFRecoTauDiscriminationByFlight(){}
    void beginEvent(const edm::Event& evt, const edm::EventSetup& es) override;
    double discriminate(const reco::PFTauRef&) override;
  private:
    edm::InputTag vertexSource_;
    edm::InputTag bsSource_;
    edm::Handle<reco::VertexCollection> vertices_;
    edm::Handle<reco::BeamSpot> beamspot_;
    const TransientTrackBuilder* builder_;
    double oneProngSig_;
    double threeProngSig_;
    bool refitPV_;
};

PFRecoTauDiscriminationByFlight::PFRecoTauDiscriminationByFlight(
    const edm::ParameterSet& pset):PFTauDiscriminationProducerBase(pset) {
  vertexSource_ = pset.getParameter<edm::InputTag>("vertexSource");
  oneProngSig_ = pset.exists("oneProngSigCut") ?
    pset.getParameter<double>("oneProngSigCut") : -1.;
  threeProngSig_ = pset.exists("threeProngSigCut") ?
    pset.getParameter<double>("threeProngSigCut") : -1.;
  refitPV_ = pset.getParameter<bool>("refitPV");
  if (refitPV_)
    bsSource_ = pset.getParameter<edm::InputTag>("beamspot");
}

void PFRecoTauDiscriminationByFlight::beginEvent(
    const edm::Event& evt, const edm::EventSetup& es) {
  evt.getByLabel(vertexSource_, vertices_);
  if (refitPV_)
    evt.getByLabel(bsSource_, beamspot_);
  edm::ESHandle<TransientTrackBuilder> transTrackBuilder;
  es.get<TransientTrackRecord>().get("TransientTrackBuilder",transTrackBuilder);
  builder_ = transTrackBuilder.product();
}

double PFRecoTauDiscriminationByFlight::discriminate(
    const reco::PFTauRef& tau) {

  KalmanVertexFitter kvf(true);
  const std::vector<reco::PFCandidatePtr>& signalTracks =
    tau->signalPFChargedHadrCands();
  std::vector<reco::TransientTrack> signalTransTracks;
  std::vector<reco::TrackRef> signalTrackPtrs;
  BOOST_FOREACH(const reco::PFCandidatePtr& pftrack, signalTracks) {
    if (pftrack->trackRef().isNonnull()) {
      signalTransTracks.push_back(
          builder_->build(pftrack->trackRef()));
      signalTrackPtrs.push_back(pftrack->trackRef());
    }
  }

  reco::Vertex pv = (*vertices_)[0];

  if (refitPV_) {
    std::vector<reco::TrackRef> pvTrackRefs;
    for (reco::Vertex::trackRef_iterator pvTrack = pv.tracks_begin();
        pvTrack != pv.tracks_end(); ++pvTrack ) {
      pvTrackRefs.push_back(pvTrack->castTo<reco::TrackRef>());
    }
    // Get PV tracks not associated to the tau
    std::sort(signalTrackPtrs.begin(), signalTrackPtrs.end());
    std::sort(pvTrackRefs.begin(), pvTrackRefs.end());
    std::vector<reco::TrackRef> uniquePVTracks;
    uniquePVTracks.reserve(pvTrackRefs.size());
    std::set_difference(pvTrackRefs.begin(), pvTrackRefs.end(),
        signalTrackPtrs.begin(), signalTrackPtrs.end(),
        std::back_inserter(uniquePVTracks));
    // Check if we need to refit
    if (uniquePVTracks.size() != pvTrackRefs.size()) {
      std::vector<reco::TransientTrack> pvTransTracks;
      // Build all our unique transient tracks in the PV
      BOOST_FOREACH(const reco::TrackRef& track, pvTrackRefs) {
        pvTransTracks.push_back(builder_->build(track));
      }
      // Refit our PV
      TransientVertex newPV = kvf.vertex(pvTransTracks, *beamspot_);
      pv = newPV;
    }
  }

  // The tau direction, to determine the sign of the IP.
  // In the case that it is a one prong, take the jet direction.
  // This may give better result due to out-of-cone stuff.
  GlobalVector direction = (tau->signalPFCands().size() == 1 ?
      GlobalVector(
          tau->jetRef()->px(), tau->jetRef()->py(), tau->jetRef()->pz()) :
      GlobalVector(tau->px(), tau->py(), tau->pz()));

  // Now figure out of we are doing a SV fit or an IP significance
  if (signalTransTracks.size() == 1) {
    reco::TransientTrack track = signalTransTracks.front();
    std::pair<bool,Measurement1D> ipsig =
      IPTools::signedTransverseImpactParameter(track, direction, pv);
    if (ipsig.first)
      return ipsig.second.significance();
    else
      return prediscriminantFailValue_;
  } else if (signalTransTracks.size() == 3) {
    // Fit the decay vertex of the three prong
    TransientVertex sv = kvf.vertex(signalTransTracks);
    // the true parameter indicates include PV errors
    Measurement1D svDist = reco::SecondaryVertex::computeDist2d(
        pv, sv, direction, true);
    double significance = svDist.significance();
    // Make sure it is a sane value
    if (significance > 40)
      significance = 40;
    if (significance < -20)
      significance = -20;
    return significance;
  } else  {
    // Weird two prong or something
    return prediscriminantFailValue_;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFRecoTauDiscriminationByFlight);
