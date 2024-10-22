
// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      METSignificance
//
/**\class METSignificance METSignificance.cc RecoMET/METAlgorithms/src/METSignificance.cc
Description: [one line class summary]
Implementation:
[Notes on implementation]
*/
//
// Original Author:  Nathan Mirman (Cornell University)
//         Created:  Thu May 30 16:39:52 CDT 2013
//
//

#include "RecoMET/METAlgorithms/interface/METSignificance.h"
#include <unordered_set>

namespace {
  struct ptr_hash {
    std::size_t operator()(const reco::CandidatePtr& k) const {
      if (k.refCore().isTransient())
        return (unsigned long)k.refCore().productPtr() ^ k.key();
      else
        return k.refCore().id().processIndex() ^ k.refCore().id().productIndex() ^ k.key();
    }
  };
}  // namespace

metsig::METSignificance::METSignificance(const edm::ParameterSet& iConfig) {
  edm::ParameterSet cfgParams = iConfig.getParameter<edm::ParameterSet>("parameters");

  double dRmatch = cfgParams.getParameter<double>("dRMatch");
  dR2match_ = dRmatch * dRmatch;

  jetThreshold_ = cfgParams.getParameter<double>("jetThreshold");
  jetEtas_ = cfgParams.getParameter<std::vector<double> >("jeta");
  jetParams_ = cfgParams.getParameter<std::vector<double> >("jpar");
  pjetParams_ = cfgParams.getParameter<std::vector<double> >("pjpar");
}

metsig::METSignificance::~METSignificance() {}

reco::METCovMatrix metsig::METSignificance::getCovariance(const edm::View<reco::Jet>& jets,
                                                          const std::vector<edm::Handle<reco::CandidateView> >& leptons,
                                                          const edm::Handle<edm::View<reco::Candidate> >& pfCandidatesH,
                                                          double rho,
                                                          JME::JetResolution& resPtObj,
                                                          JME::JetResolution& resPhiObj,
                                                          JME::JetResolutionScaleFactor& resSFObj,
                                                          bool isRealData,
                                                          double& sumPtUnclustered,
                                                          edm::ValueMap<float> const* weights) {
  //pfcandidates
  const edm::View<reco::Candidate>& pfCandidates = *pfCandidatesH;

  // metsig covariance
  double cov_xx = 0;
  double cov_xy = 0;
  double cov_yy = 0;

  // for lepton and jet subtraction
  std::unordered_set<reco::CandidatePtr, ptr_hash> footprint;

  // subtract leptons out of sumPtUnclustered
  for (const auto& lep_i : leptons) {
    for (const auto& lep : lep_i->ptrs()) {
      if (lep->pt() > 10) {
        for (unsigned int n = 0; n < lep->numberOfSourceCandidatePtrs(); n++)
          footprint.insert(lep->sourceCandidatePtr(n));
      }
    }
  }

  std::vector<bool> cleanedJets(jets.size(), false);
  std::transform(jets.begin(), jets.end(), cleanedJets.begin(), [this, &leptons](auto const& jet) -> bool {
    return cleanJet(jet, leptons);
  });
  // subtract jets out of sumPtUnclustered
  auto iCleaned = cleanedJets.begin();
  for (const auto& jet : jets) {
    // disambiguate jets and leptons
    if (!(*iCleaned++))
      continue;
    for (unsigned int n = 0; n < jet.numberOfSourceCandidatePtrs(); n++) {
      footprint.insert(jet.sourceCandidatePtr(n));
    }
  }

  // calculate sumPtUnclustered
  for (size_t i = 0; i < pfCandidates.size(); ++i) {
    // check if candidate exists in a lepton or jet
    bool cleancand = true;
    if (footprint.find(pfCandidates.ptrAt(i)) == footprint.end()) {
      float weight = (weights != nullptr) ? (*weights)[pfCandidates.ptrAt(i)] : 1.0;
      //dP4 recovery
      for (const auto& it : footprint) {
        if (it.isNonnull() && it.isAvailable() && (reco::deltaR2(*it, pfCandidates[i]) < 0.00000025)) {
          cleancand = false;
          break;
        }
      }
      // if not, add to sumPtUnclustered
      if (cleancand) {
        sumPtUnclustered += pfCandidates[i].pt() * weight;
      }
    }
  }

  // add jets to metsig covariance matrix and subtract them from sumPtUnclustered
  iCleaned = cleanedJets.begin();
  for (const auto& jet : jets) {
    // disambiguate jets and leptons
    if (!(*iCleaned++))
      continue;

    double jpt = jet.pt();

    // split into high-pt and low-pt sector
    if (jpt > jetThreshold_) {
      // high-pt jets enter into the covariance matrix via JER

      double jeta = jet.eta();
      double feta = std::abs(jeta);
      double c = jet.px() / jpt;
      double s = jet.py() / jpt;

      JME::JetParameters parameters;
      parameters.setJetPt(jpt).setJetEta(jeta).setRho(rho);

      // jet energy resolutions
      double sigmapt = resPtObj.getResolution(parameters);
      double sigmaphi = resPhiObj.getResolution(parameters);
      // SF not needed since is already embedded in the sigma in the dataGlobalTag
      //      double sigmaSF = isRealData ? resSFObj.getScaleFactor(parameters) : 1.0;

      double scale = 0;
      if (feta < jetEtas_[0])
        scale = jetParams_[0];
      else if (feta < jetEtas_[1])
        scale = jetParams_[1];
      else if (feta < jetEtas_[2])
        scale = jetParams_[2];
      else if (feta < jetEtas_[3])
        scale = jetParams_[3];
      else
        scale = jetParams_[4];

      //         double dpt = sigmaSF*scale*jpt*sigmapt;
      double dpt = scale * jpt * sigmapt;
      double dph = jpt * sigmaphi;

      cov_xx += dpt * dpt * c * c + dph * dph * s * s;
      cov_xy += (dpt * dpt - dph * dph) * c * s;
      cov_yy += dph * dph * c * c + dpt * dpt * s * s;

    } else {
      // add the (corrected) jet to the sumPtUnclustered
      sumPtUnclustered += jpt;
    }
  }

  //protection against unphysical events
  if (sumPtUnclustered < 0)
    sumPtUnclustered = 0;

  // add pseudo-jet to metsig covariance matrix
  double pseudoJetCov = pjetParams_[0] * pjetParams_[0] + pjetParams_[1] * pjetParams_[1] * sumPtUnclustered;
  cov_xx += pseudoJetCov;
  cov_yy += pseudoJetCov;

  reco::METCovMatrix cov;
  cov(0, 0) = cov_xx;
  cov(1, 0) = cov_xy;
  cov(0, 1) = cov_xy;
  cov(1, 1) = cov_yy;

  return cov;
}

double metsig::METSignificance::getSignificance(const reco::METCovMatrix& cov, const reco::MET& met) {
  // covariance matrix determinant
  double det = cov(0, 0) * cov(1, 1) - cov(0, 1) * cov(1, 0);

  // invert matrix
  double ncov_xx = cov(1, 1) / det;
  double ncov_xy = -cov(0, 1) / det;
  double ncov_yy = cov(0, 0) / det;

  // product of met and inverse of covariance
  double sig = met.px() * met.px() * ncov_xx + 2 * met.px() * met.py() * ncov_xy + met.py() * met.py() * ncov_yy;

  return sig;
}

bool metsig::METSignificance::cleanJet(const reco::Jet& jet,
                                       const std::vector<edm::Handle<reco::CandidateView> >& leptons) {
  for (const auto& lep_i : leptons) {
    for (const auto& lep : *lep_i) {
      if (reco::deltaR2(lep, jet) < dR2match_) {
        return false;
      }
    }
  }
  return true;
}
