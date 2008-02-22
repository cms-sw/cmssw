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

//     float tZ = it->vz();
//     float tD0 = fabs(it->d0());
//     float tD0Cor = fabs(it->dxy(thePars.beamPoint));
//     float tEta = it->eta();
//     float tPhi = it->phi();
//     uint tHits = it->numberOfValidHits();
//     float tChi2Ndof = it->normalizedChi2();
//     float tChi2Prob = ChiSquaredProbability(it->chi2(), it->ndof());
//     float tPt = it->pt();

//     LogTrace(metname)<<"Tk vz: "<<tZ
// 		     <<",  d0: "<<tD0
// 		     <<",  d0wrtBeam: "<<tD0Cor
// 		     <<", eta: "<<tEta
// 		     <<", phi: "<<tPhi
// 		     <<", nHits: "<<tHits
// 		     <<", chi2Norm: "<<tChi2Ndof
// 		     <<", chi2Prob: "<<tChi2Prob
// 		     <<std::endl;

    //! pick/read variables in order to cut down on unnecessary calls
    //! someone will have some fun reading the log if Debug is on
    //! the biggest reason is the numberOfValidHits call (the rest are not as costly)

    float tZ = it->vz(); 
    LogTrace(metname)<<"Tk vz: "<<tZ;
    if ( !thePars.zRange.inside( tZ ) ) continue; 

    float tPt = it->pt();
    LogTrace(metname)<<",  pt: "<<tPt;
    if ( tPt < thePars.ptMin ) continue;

    float tD0 = fabs(it->d0());  
    float tD0Cor = fabs(it->dxy(thePars.beamPoint));
    LogTrace(metname)<<",  d0: "<<tD0
 		     <<",  d0wrtBeam: "<<tD0Cor;
    if ( !thePars.rRange.inside( tD0Cor) ) continue;


    float tEta = it->eta();
    float tPhi = it->phi();
    LogTrace(metname)<<", eta: "<<tEta
 		     <<", phi: "<<tPhi;
    if ( thePars.dir.deltaR( Direction(tEta, tPhi) ) > thePars.drMax ) continue;

    uint tHits = it->numberOfValidHits();
    LogTrace(metname)<<", nHits: "<<tHits;
    if ( tHits < thePars.nHitsMin ) continue;


    float tChi2Ndof = it->normalizedChi2();
    LogTrace(metname)<<", chi2Norm: "<<tChi2Ndof;
    if ( tChi2Ndof > thePars.chi2NdofMax ) continue;


    float tChi2Prob = ChiSquaredProbability(it->chi2(), it->ndof());
    LogTrace(metname)<<", chi2Prob: "<<tChi2Prob<<std::endl;
    if ( tChi2Prob < thePars.chi2ProbMin ) continue;


    LogTrace(metname)<<" ..... accepted"<<std::endl;
    result.push_back(&*it);
  } 
  return result;
}
