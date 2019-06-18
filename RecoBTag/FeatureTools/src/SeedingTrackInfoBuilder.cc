#include "RecoBTag/FeatureTools/interface/SeedingTrackInfoBuilder.h"
#include "DataFormats/GeometrySurface/interface/Line.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "RecoBTag/TrackProbability/interface/HistogramProbabilityEstimator.h"

namespace btagbtvdeep {

  SeedingTrackInfoBuilder::SeedingTrackInfoBuilder()
      : pt_(0),
        eta_(0),
        phi_(0),
        mass_(0),
        dz_(0),
        dxy_(0),
        ip3D_(0),
        sip3D_(0),
        ip2D_(0),
        sip2D_(0),
        ip3D_signed_(0),
        sip3D_signed_(0),
        ip2D_signed_(0),
        sip2D_signed_(0),
        chi2reduced_(0),
        nPixelHits_(0),
        nHits_(0),
        jetAxisDistance_(0),
        jetAxisDlength_(0),
        trackProbability3D_(0),
        trackProbability2D_(0) {}

  void SeedingTrackInfoBuilder::buildSeedingTrackInfo(const reco::TransientTrack* it,
                                                      const reco::Vertex& pv,
                                                      const reco::Jet& jet, /*GlobalVector jetdirection,*/
                                                      float mass,
                                                      const std::pair<bool, Measurement1D>& ip,
                                                      const std::pair<bool, Measurement1D>& ip2d,
                                                      float jet_distance,
                                                      float jaxis_dlength,
                                                      HistogramProbabilityEstimator* m_probabilityEstimator,
                                                      bool m_computeProbabilities = false) {
    GlobalPoint pvp(pv.x(), pv.y(), pv.z());
    GlobalVector jetdirection(jet.px(), jet.py(), jet.pz());

    auto const& aTrack = it->track();

    pt_ = aTrack.pt();
    eta_ = aTrack.eta();
    phi_ = aTrack.phi();
    dz_ = aTrack.dz(pv.position());
    dxy_ = aTrack.dxy(pv.position());
    mass_ = mass;

    std::pair<bool, Measurement1D> ipSigned = IPTools::signedImpactParameter3D(*it, jetdirection, pv);
    std::pair<bool, Measurement1D> ip2dSigned = IPTools::signedTransverseImpactParameter(*it, jetdirection, pv);

    ip3D_ = ip.second.value();
    sip3D_ = ip.second.significance();
    ip2D_ = ip2d.second.value();
    sip2D_ = ip2d.second.significance();
    ip3D_signed_ = ipSigned.second.value();
    sip3D_signed_ = ipSigned.second.significance();
    ip2D_signed_ = ip2dSigned.second.value();
    sip2D_signed_ = ip2dSigned.second.significance();

    chi2reduced_ = aTrack.normalizedChi2();
    nPixelHits_ = aTrack.hitPattern().numberOfValidPixelHits();
    nHits_ = aTrack.hitPattern().numberOfValidHits();

    jetAxisDistance_ = std::fabs(jet_distance);
    jetAxisDlength_ = jaxis_dlength;

    trackProbability3D_ = 0.5;
    trackProbability2D_ = 0.5;

    if (m_computeProbabilities) {
      //probability with 3D ip
      std::pair<bool, double> probability =
          m_probabilityEstimator->probability(false, 0, ip.second.significance(), aTrack, jet, pv);
      double prob3D = (probability.first ? probability.second : -1.);

      //probability with 2D ip
      probability = m_probabilityEstimator->probability(false, 1, ip2d.second.significance(), aTrack, jet, pv);
      double prob2D = (probability.first ? probability.second : -1.);

      trackProbability3D_ = prob3D;
      trackProbability2D_ = prob2D;
    }

    if (!edm::isFinite(trackProbability3D_))
      trackProbability3D_ = 0.5;
    if (!edm::isFinite(trackProbability2D_))
      trackProbability2D_ = 0.5;
  }

}  // namespace btagbtvdeep
