#include "DataFormats/MuonReco/interface/MuonWithMatchInfo.h"
using namespace reco;

MuonWithMatchInfo::MuonWithMatchInfo(  Charge q, const LorentzVector & p4, const Point & vtx ) : 
   Muon( q, p4, vtx ) 
{}

int MuonWithMatchInfo::numberOfMatches() const
{
   int matches(0);
   for(std::vector<MuonChamberMatch>::const_iterator chamberMatch = muMatches_.begin();
       chamberMatch != muMatches_.end(); chamberMatch++)
     if (! chamberMatch->segmentMatches.empty() ) matches++;
   return matches;
}

float MuonWithMatchInfo::dX(uint i) const
{
   if (muMatches_.size()>i && ! muMatches_[i].segmentMatches.empty() )
     return muMatches_[i].segmentMatches.front().x-muMatches_[i].x;
   else
     return 999999.;
}

float MuonWithMatchInfo::dY(uint i) const
{
   if (muMatches_.size()>i && ! muMatches_[i].segmentMatches.empty() )
     return muMatches_[i].segmentMatches.front().y-muMatches_[i].y;
   else
     return 999999.;
}

float MuonWithMatchInfo::dXErr(uint i) const
{
   if (muMatches_.size()>i && ! muMatches_[i].segmentMatches.empty() )
     return sqrt( pow(muMatches_[i].segmentMatches.front().xErr,2) +
		  pow(muMatches_[i].xErr,2) );
   else
     return 999.;
}

float MuonWithMatchInfo::dYErr(uint i) const
{
   if (muMatches_.size()>i && ! muMatches_[i].segmentMatches.empty() )
     return sqrt( pow(muMatches_[i].segmentMatches.front().yErr,2) + 
		  pow(muMatches_[i].yErr,2) );
   else
     return 999.;
}



