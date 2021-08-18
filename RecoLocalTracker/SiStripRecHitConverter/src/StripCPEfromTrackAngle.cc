#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"

#include "vdt/vdtMath.h"

StripCPEfromTrackAngle::StripCPEfromTrackAngle(edm::ParameterSet& conf,
                                               const MagneticField& mag,
                                               const TrackerGeometry& geom,
                                               const SiStripLorentzAngle& lorentz,
                                               const SiStripBackPlaneCorrection& backPlaneCorrection,
                                               const SiStripConfObject& confObj,
                                               const SiStripLatency& latency)
    : StripCPE(conf, mag, geom, lorentz, backPlaneCorrection, confObj, latency),
      useLegacyError(conf.existsAs<bool>("useLegacyError") ? conf.getParameter<bool>("useLegacyError") : true),
      maxChgOneMIP(conf.existsAs<float>("maxChgOneMIP") ? conf.getParameter<double>("maxChgOneMIP") : -6000.),
      m_algo(useLegacyError ? Algo::legacy : (maxChgOneMIP < 0 ? Algo::mergeCK : Algo::chargeCK)) {
  mLC_P[0] = conf.existsAs<double>("mLC_P0") ? conf.getParameter<double>("mLC_P0") : -.326;
  mLC_P[1] = conf.existsAs<double>("mLC_P1") ? conf.getParameter<double>("mLC_P1") : .618;
  mLC_P[2] = conf.existsAs<double>("mLC_P2") ? conf.getParameter<double>("mLC_P2") : .300;

  mHC_P[SiStripDetId::TIB - 3][0] = conf.existsAs<double>("mTIB_P0") ? conf.getParameter<double>("mTIB_P0") : -.742;
  mHC_P[SiStripDetId::TIB - 3][1] = conf.existsAs<double>("mTIB_P1") ? conf.getParameter<double>("mTIB_P1") : .202;
  mHC_P[SiStripDetId::TID - 3][0] = conf.existsAs<double>("mTID_P0") ? conf.getParameter<double>("mTID_P0") : -1.026;
  mHC_P[SiStripDetId::TID - 3][1] = conf.existsAs<double>("mTID_P1") ? conf.getParameter<double>("mTID_P1") : .253;
  mHC_P[SiStripDetId::TOB - 3][0] = conf.existsAs<double>("mTOB_P0") ? conf.getParameter<double>("mTOB_P0") : -1.427;
  mHC_P[SiStripDetId::TOB - 3][1] = conf.existsAs<double>("mTOB_P1") ? conf.getParameter<double>("mTOB_P1") : .433;
  mHC_P[SiStripDetId::TEC - 3][0] = conf.existsAs<double>("mTEC_P0") ? conf.getParameter<double>("mTEC_P0") : -1.885;
  mHC_P[SiStripDetId::TEC - 3][1] = conf.existsAs<double>("mTEC_P1") ? conf.getParameter<double>("mTEC_P1") : .471;
}

float StripCPEfromTrackAngle::stripErrorSquared(const unsigned N,
                                                const float uProj,
                                                const SiStripDetId::SubDetector loc) const {
  auto fun = [&](float x) -> float { return mLC_P[0] * x * vdt::fast_expf(-x * mLC_P[1]) + mLC_P[2]; };
  auto uerr = (N <= 4) ? fun(uProj) : mHC_P[loc - 3][0] + float(N) * mHC_P[loc - 3][1];
  return uerr * uerr;
}

float StripCPEfromTrackAngle::legacyStripErrorSquared(const unsigned N, const float uProj) const {
  if UNLIKELY ((float(N) - uProj) > 3.5f)
    return float(N * N) / 12.f;
  else {
    static constexpr float P1 = -0.339;
    static constexpr float P2 = 0.90;
    static constexpr float P3 = 0.279;
    const float uerr = P1 * uProj * vdt::fast_expf(-uProj * P2) + P3;
    return uerr * uerr;
  }
}

void StripCPEfromTrackAngle::localParameters(AClusters const& clusters,
                                             ALocalValues& retValues,
                                             const GeomDetUnit& det,
                                             const LocalTrajectoryParameters& ltp) const {
  auto const& par = getAlgoParam(det, ltp);
  auto const& p = par.p;
  auto loc = par.loc;
  auto corr = par.corr;
  auto afp = par.afullProjection;

  auto fill = [&](unsigned int i, float uerr2) {
    const float strip = clusters[i]->barycenter() + corr;
    retValues[i].first = p.topology->localPosition(strip, ltp.vector());
    retValues[i].second = p.topology->localError(strip, uerr2, ltp.vector());
  };

  switch (m_algo) {
    case Algo::chargeCK:
      for (auto i = 0U; i < clusters.size(); ++i) {
        auto dQdx = siStripClusterTools::chargePerCM(*clusters[i], ltp, p.invThickness);
        auto N = clusters[i]->amplitudes().size();
        auto uerr2 = dQdx > maxChgOneMIP ? legacyStripErrorSquared(N, afp) : stripErrorSquared(N, afp, loc);
        fill(i, uerr2);
      }
      break;
    case Algo::legacy:
      for (auto i = 0U; i < clusters.size(); ++i) {
        auto N = clusters[i]->amplitudes().size();
        auto uerr2 = legacyStripErrorSquared(N, afp);
        fill(i, uerr2);
      }
      break;
    case Algo::mergeCK:
      for (auto i = 0U; i < clusters.size(); ++i) {
        auto N = clusters[i]->amplitudes().size();
        auto uerr2 = clusters[i]->isMerged() ? legacyStripErrorSquared(N, afp) : stripErrorSquared(N, afp, loc);
        fill(i, uerr2);
      }
      break;
  }
}

StripClusterParameterEstimator::LocalValues StripCPEfromTrackAngle::localParameters(const SiStripCluster& cluster,
                                                                                    AlgoParam const& par) const {
  auto const& p = par.p;
  auto const& ltp = par.ltp;
  auto loc = par.loc;
  auto corr = par.corr;
  auto afp = par.afullProjection;

  float uerr2 = 0;

  auto N = cluster.amplitudes().size();

  switch (m_algo) {
    case Algo::chargeCK: {
      auto dQdx = siStripClusterTools::chargePerCM(cluster, ltp, p.invThickness);
      uerr2 = dQdx > maxChgOneMIP ? legacyStripErrorSquared(N, afp) : stripErrorSquared(N, afp, loc);
    } break;
    case Algo::legacy:
      uerr2 = legacyStripErrorSquared(N, afp);
      break;
    case Algo::mergeCK:
      uerr2 = cluster.isMerged() ? legacyStripErrorSquared(N, afp) : stripErrorSquared(N, afp, loc);
      break;
  }

  const float strip = cluster.barycenter() + corr;

  return std::make_pair(p.topology->localPosition(strip, ltp.vector()),
                        p.topology->localError(strip, uerr2, ltp.vector()));
}

StripClusterParameterEstimator::LocalValues StripCPEfromTrackAngle::localParameters(
    const SiStripCluster& cluster, const GeomDetUnit& det, const LocalTrajectoryParameters& ltp) const {
  auto const& par = getAlgoParam(det, ltp);
  return localParameters(cluster, par);
}
