#ifndef __RecoParticleFlow_Benchmark_Matchers__
#define __RecoParticleFlow_Benchmark_Matchers__

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <iostream>
#include <vector>

namespace PFB {

  template <typename C, typename M>
  void match(const C &candCollection,
             const M &matchedCandCollection,
             std::vector<int> &matchIndices,
             bool matchCharge = false,
             float dRMax = -1) {
    // compute distance to each candidate in the matchedCandCollection.

    float dR2Max = 0;
    if (dRMax > 0)
      dR2Max = dRMax * dRMax;

    matchIndices.clear();
    matchIndices.resize(candCollection.size(), -1);

    for (unsigned i = 0; i < candCollection.size(); ++i) {
      static const double bigNumber = 1e14;
      double dR2min = bigNumber;
      int jMin = -1;
      for (unsigned jm = 0; jm < matchedCandCollection.size(); ++jm) {
        if (matchCharge && candCollection[i].charge() != matchedCandCollection[jm].charge())
          continue;

        double dR2 = reco::deltaR2(candCollection[i], matchedCandCollection[jm]);
        if (dR2 < dR2min) {
          dR2min = dR2;
          jMin = jm;
        }
      }

      if ((dR2Max > 0 && dR2min < dR2Max) || dRMax <= 0) {
        matchIndices[i] = jMin;
        /* 	std::cout<<"match "<<dR2min<<std::endl;  */
      }
      // store the closest match, no cut on deltaR.
    }
  }

  // needed by muon matching
  // template< typename C, typename M, typename MM>
  template <typename C, typename M>
  void match(const C &candCollection,
             const M &matchedCandCollection,
             std::vector<int> &matchIndices,
             const edm::ParameterSet &parameterSet,
             // const MM& muonMatchedCandCollection,
             edm::View<reco::Muon> muonMatchedCandCollection,
             bool matchCharge = false,
             float dRMax = -1) {
    // compute distance to each candidate in the matchedCandCollection.

    float dR2Max = 0;
    if (dRMax > 0)
      dR2Max = dRMax * dRMax;

    matchIndices.clear();
    matchIndices.resize(candCollection.size(), -1);

    for (unsigned i = 0; i < candCollection.size(); ++i) {
      static const double bigNumber = 1e14;
      double dR2min = bigNumber;
      int jMin = -1;
      for (unsigned jm = 0; jm < matchedCandCollection.size(); ++jm) {
        if (parameterSet.getParameter<bool>("slimmedLikeSelection")) {
          if (!(muonMatchedCandCollection[jm].pt() > parameterSet.getParameter<double>("ptBase") ||
                muonMatchedCandCollection[jm].isPFMuon() ||
                (muonMatchedCandCollection[jm].pt() > parameterSet.getParameter<double>("ptNotPF") &&
                 (muonMatchedCandCollection[jm].isGlobalMuon() || muonMatchedCandCollection[jm].isStandAloneMuon() ||
                  muonMatchedCandCollection[jm].numberOfMatches() > 0 ||
                  muon::isGoodMuon(muonMatchedCandCollection[jm], muon::RPCMuLoose)))))
            continue;
        }

        if (matchCharge && candCollection[i].charge() != matchedCandCollection[jm].charge())
          continue;

        double dR2 = reco::deltaR2(candCollection[i], matchedCandCollection[jm]);
        if (dR2 < dR2min) {
          dR2min = dR2;
          jMin = jm;
        }
      }

      if ((dR2Max > 0 && dR2min < dR2Max) || dRMax <= 0) {
        matchIndices[i] = jMin;
        /* 	std::cout<<"match "<<dR2min<<std::endl;  */
      }
      // store the closest match, no cut on deltaR.
    }
  }

}  // namespace PFB

#endif
