#ifndef RecoSelectors_GenParticleCustomSelector_h
#define RecoSelectors_GenParticleCustomSelector_h
/* \class GenParticleCustomSelector
 *
 * \author Giuseppe Cerati, UCSD
 *
 */

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class GenParticleCustomSelector {
public:
  GenParticleCustomSelector() {}
  GenParticleCustomSelector(double ptMin,
                            double minRapidity,
                            double maxRapidity,
                            double tip,
                            double lip,
                            bool chargedOnly,
                            int status,
                            const std::vector<int>& pdgId = std::vector<int>(),
                            bool invertRapidityCut = false,
                            double minPhi = -3.2,
                            double maxPhi = 3.2)
      : ptMin_(ptMin),
        minRapidity_(minRapidity),
        maxRapidity_(maxRapidity),
        meanPhi_((minPhi + maxPhi) / 2.),
        rangePhi_((maxPhi - minPhi) / 2.),
        tip_(tip),
        lip_(lip),
        chargedOnly_(chargedOnly),
        status_(status),
        pdgId_(pdgId),
        invertRapidityCut_(invertRapidityCut) {
    if (minPhi >= maxPhi) {
      throw cms::Exception("Configuration")
          << "GenParticleCustomSelector: minPhi (" << minPhi << ") must be smaller than maxPhi (" << maxPhi
          << "). The range is constructed from minPhi to maxPhi around their "
             "average.";
    }
    if (minPhi >= M_PI) {
      throw cms::Exception("Configuration") << "GenParticleCustomSelector: minPhi (" << minPhi
                                            << ") must be smaller than PI. The range is constructed from minPhi "
                                               "to maxPhi around their average.";
    }
    if (maxPhi <= -M_PI) {
      throw cms::Exception("Configuration") << "GenParticleCustomSelector: maxPhi (" << maxPhi
                                            << ") must be larger than -PI. The range is constructed from minPhi "
                                               "to maxPhi around their average.";
    }
  }

  /// Operator() performs the selection: e.g. if (tPSelector(tp)) {...}
  bool operator()(const reco::GenParticle& tp) const {
    if (chargedOnly_ && tp.charge() == 0)
      return false;  //select only if charge!=0
    bool testId = false;
    unsigned int idSize = pdgId_.size();
    if (idSize == 0)
      testId = true;
    else
      for (unsigned int it = 0; it != idSize; ++it) {
        if (tp.pdgId() == pdgId_[it])
          testId = true;
      }

    auto etaOk = [&](const reco::GenParticle& p) -> bool {
      float eta = p.eta();
      if (!invertRapidityCut_)
        return (eta >= minRapidity_) && (eta <= maxRapidity_);
      else
        return (eta < minRapidity_ || eta > maxRapidity_);
    };
    auto phiOk = [&](const reco::GenParticle& p) {
      float dphi = deltaPhi(atan2f(p.py(), p.px()), meanPhi_);
      return dphi >= -rangePhi_ && dphi <= rangePhi_;
    };
    auto ptOk = [&](const reco::GenParticle& p) {
      double pt = p.pt();
      return pt >= ptMin_;
    };

    return (ptOk(tp) && etaOk(tp) && phiOk(tp) && sqrt(tp.vertex().perp2()) <= tip_ && fabs(tp.vertex().z()) <= lip_ &&
            tp.status() == status_ && testId);
  }

private:
  double ptMin_;
  double minRapidity_;
  double maxRapidity_;
  float meanPhi_;
  float rangePhi_;
  double tip_;
  double lip_;
  bool chargedOnly_;
  int status_;
  std::vector<int> pdgId_;
  bool invertRapidityCut_;
};

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"

namespace reco {
  namespace modules {

    template <>
    struct ParameterAdapter<GenParticleCustomSelector> {
      static GenParticleCustomSelector make(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC) {
        return make(cfg);
      }

      static GenParticleCustomSelector make(const edm::ParameterSet& cfg) {
        return GenParticleCustomSelector(cfg.getParameter<double>("ptMin"),
                                         cfg.getParameter<double>("minRapidity"),
                                         cfg.getParameter<double>("maxRapidity"),
                                         cfg.getParameter<double>("tip"),
                                         cfg.getParameter<double>("lip"),
                                         cfg.getParameter<bool>("chargedOnly"),
                                         cfg.getParameter<int>("status"),
                                         cfg.getParameter<std::vector<int> >("pdgId"),
                                         cfg.getParameter<bool>("invertRapidityCut"),
                                         cfg.getParameter<double>("minPhi"),
                                         cfg.getParameter<double>("maxPhi"));
      }
    };

  }  // namespace modules
}  // namespace reco

#endif
