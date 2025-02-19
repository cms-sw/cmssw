#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTrackSelector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

using namespace egammaisolation;
using namespace reco;

EgammaTrackSelector::result_type EgammaTrackSelector::operator()(const EgammaTrackSelector::input_type & tracks) const
{
  static std::string metname = "EgammaIsolationAlgos|EgammaTrackSelector";
  result_type result;
  for (input_type::const_iterator it = tracks.begin(); it != tracks.end(); it++) {

    //! pick/read variables in order to cut down on unnecessary calls
    //! someone will have some fun reading the log if Debug is on
    //! the biggest reason is the numberOfValidHits call (the rest are not as costly)

    float tZ;
    switch(thePars.dzOption) {
        case dz : tZ = it->dz();                  break;
        case vz : tZ = it->vz();                  break;
        case bs : tZ = it->dz(thePars.beamPoint); break;
        default : tZ = it->vz();                  break;
    }

    float tPt = it->pt();
    //float tD0 = fabs(it->d0());  //currently not used.  
    float tD0Cor = fabs(it->dxy(thePars.beamPoint));
    float tEta = it->eta();
    float tPhi = it->phi();
    float tChi2Ndof = it->normalizedChi2();
  
    //! access to the remaining vars is slow

    if ( !thePars.zRange.inside( tZ ) ) continue; 
    if ( tPt < thePars.ptMin ) continue;
    if ( !thePars.rRange.inside( tD0Cor) ) continue;
    if ( thePars.dir.deltaR( reco::isodeposit::Direction(tEta, tPhi) ) > thePars.drMax ) continue;
    if ( tChi2Ndof > thePars.chi2NdofMax ) continue;

    //! skip if min Hits == 0; assumes any track has at least one valid hit
    if (thePars.nHitsMin > 0 ){
      unsigned int tHits = it->numberOfValidHits();
      if ( tHits < thePars.nHitsMin ) continue;
    }

    //! similarly here
    if(thePars.chi2ProbMin > 0){
      float tChi2Prob = ChiSquaredProbability(it->chi2(), it->ndof());
      if ( tChi2Prob < thePars.chi2ProbMin ) continue;
    }
    result.push_back(&*it);
  } 
  return result;
}
