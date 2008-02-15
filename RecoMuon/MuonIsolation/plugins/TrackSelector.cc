#include "TrackSelector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

using namespace muonisolation;
using namespace reco;

TrackSelector::result_type TrackSelector::operator()(const TrackSelector::input_type & tracks) const
{
  static std::string metname = "MuonIsolation|TrackSelector";
  result_type result;
  for (input_type::const_iterator it = tracks.begin(); it != tracks.end(); it++) {
    float tZ = it->vz();
    float tD0 = fabs(it->d0());
    float tD0Cor = fabs(it->dxy(thePars.beamPoint));
    float tEta = it->eta();
    float tPhi = it->phi();
    uint tHits = it->numberOfValidHits();
    float tChi2Ndof = it->normalizedChi2();
    float tChi2Prob = ChiSquaredProbability(it->chi2(), it->ndof());
    float tPt = it->pt();

    LogTrace(metname)<<"Tk vz: "<<tZ
		     <<",  d0: "<<tD0
		     <<",  d0wrtBeam: "<<tD0Cor
		     <<", eta: "<<tEta
		     <<", phi: "<<tPhi
		     <<", nHits: "<<tHits
		     <<", chi2Norm: "<<tChi2Ndof
		     <<", chi2Prob: "<<tChi2Prob
		     <<std::endl;

    if ( !thePars.zRange.inside( tZ ) ) continue; 
    if ( !thePars.rRange.inside( tD0Cor) ) continue;
    if ( thePars.dir.deltaR( Direction(tEta, tPhi) ) > thePars.drMax ) continue;
    if ( tHits < thePars.nHitsMin ) continue;
    if ( tChi2Ndof > thePars.chi2NdofMax ) continue;
    if ( tChi2Prob < thePars.chi2ProbMin ) continue;
    if ( tPt < thePars.ptMin ) continue;

    LogTrace(metname)<<" ..... accepted"<<std::endl;
    result.push_back(&*it);
  } 
  return result;
}
