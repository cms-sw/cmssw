#include "TrackSelector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

using namespace muonisolation;
using namespace reco;

TrackSelector::result_type TrackSelector::operator()(const TrackSelector::input_type& tracks) const {
  static const std::string metname = "MuonIsolation|TrackSelector";
  result_type result;
  auto const dr2Max = thePars.drMax * thePars.drMax;
  for (auto const& tk : tracks) {
    //! pick/read variables in order to cut down on unnecessary calls
    //! someone will have some fun reading the log if Debug is on
    //! the biggest reason is the numberOfValidHits call (the rest are not as costly)

    float tZ = tk.vz();
    float tPt = tk.pt();
    float tD0 = fabs(tk.d0());
    float tD0Cor = fabs(tk.dxy(thePars.beamPoint));
    float tChi2Ndof = tk.normalizedChi2();
    LogTrace(metname) << "Tk vz: " << tZ << ",  pt: " << tPt << ",  d0: " << tD0 << ",  d0wrtBeam: " << tD0Cor
                      << ", eta: " << tk.eta() << ", phi: " << tk.phi() << ", chi2Norm: " << tChi2Ndof;
    //! access to the remaining vars is slow

    if (!thePars.zRange.inside(tZ))
      continue;
    if (tPt < thePars.ptMin)
      continue;
    if (!thePars.rRange.inside(tD0Cor))
      continue;
    if (tChi2Ndof > thePars.chi2NdofMax)
      continue;
    float tEta = tk.eta();
    float tPhi = tk.phi();
    if (thePars.dir.deltaR2(reco::isodeposit::Direction(tEta, tPhi)) > dr2Max)
      continue;

    //! skip if min Hits == 0; assumes any track has at least one valid hit
    if (thePars.nHitsMin > 0) {
      unsigned int tHits = tk.numberOfValidHits();
      LogTrace(metname) << ", nHits: " << tHits;
      if (tHits < thePars.nHitsMin)
        continue;
    }

    //! similarly here
    if (thePars.chi2ProbMin > 0) {
      float tChi2Prob = ChiSquaredProbability(tk.chi2(), tk.ndof());
      LogTrace(metname) << ", chi2Prob: " << tChi2Prob << std::endl;
      if (tChi2Prob < thePars.chi2ProbMin)
        continue;
    }

    LogTrace(metname) << " ..... accepted" << std::endl;
    result.push_back(&tk);
  }
  return result;
}
