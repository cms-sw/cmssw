#include "TrackSelector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


using namespace muonisolation;
using namespace reco;
TrackSelector::TrackSelector(const Range & z, float r, const Direction & dir, float drMax)
  : theZ(z), theR(Range(0.,r)), theDir(dir), theDR_Max(drMax)
{ } 

TrackSelector::result_type TrackSelector::operator()(const TrackSelector::input_type & tracks) const
{
  static std::string metname = "MuonIsolation|TrackSelector";
  result_type result;
  for (input_type::const_iterator it = tracks.begin(); it != tracks.end(); it++) {
    LogTrace(metname)<<"Tk vz: "<<it->vz()
		     <<",  d0: "<<fabs(it->d0())
		     <<", eta: "<<it->eta()
		     <<", phi: "<<it->phi()
		     <<std::endl;
    if ( !theZ.inside( (*it).vz() ) ) continue; 
    if ( !theR.inside( fabs((*it).d0()) ) ) continue;
    if ( theDir.deltaR( Direction(it->eta(), it->phi()) ) > theDR_Max ) continue;
    LogTrace(metname)<<" ..... accepted"<<std::endl;
    result.push_back(&*it);
  } 
  return result;
}
