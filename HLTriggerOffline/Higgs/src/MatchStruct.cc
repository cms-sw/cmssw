#ifndef HLTRIGGEROFFLINE_HIGGS_MATCHSTRUCT_CC
#define HLTRIGGEROFFLINE_HIGGS_MATCHSTRUCT_CC

/** \class MatchStruct
 *  Generate histograms for trigger efficiencies Higgs related
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/HiggsWGHLTValidate
 *
 *  $Date: 2012/03/16 01:55:32 $
 *  $Revision: 1.1 $
 *  \author  J. Duarte Campderros
 *
 */

//#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
// FIXME: Is there any way to avoid this!!??? Search for inheritance of track
#include "DataFormats/TrackReco/interface/Track.h"

#include<vector>

// Matching structure: helper structure to match gen/reco candidates with
// hlt trigger objects
struct MatchStruct 
{
	unsigned int objType;
	const reco::Candidate * candBase;
	const reco::Track     * trackCandBase;
	MatchStruct() 
	{
		candBase       = 0;
		trackCandBase  = 0;
	}
	MatchStruct(const reco::Candidate * cand, const unsigned int & obj) :
		objType(obj),
		candBase(cand),
		trackCandBase(0)
	{
	}
	MatchStruct(const reco::Track * cand, const unsigned int & obj) :
		objType(obj),
		candBase(0),
		trackCandBase(cand)
	{
	}
	bool operator<(MatchStruct match) 
	{      
		//FIXME: 
		if( candBase != 0 )
		{
			if( match.candBase != 0 )
			{
				return candBase->pt() < match.candBase->pt();
			}
			else if( match.trackCandBase != 0 )
			{
				return candBase->pt() < match.trackCandBase->pt();
			}
		}
		else if( trackCandBase != 0 )
		{
			if( match.candBase != 0 )
			{
				return trackCandBase->pt() < match.candBase->pt();
			}
			else if( match.trackCandBase != 0 )
			{
				return trackCandBase->pt() < match.trackCandBase->pt();
			}
		}
				
		//return candBase->pt() < match.candBase->pt();
	}
	bool operator>(MatchStruct match) 
	{
		if( candBase != 0 )
		{
			if( match.candBase != 0 )
			{
				return candBase->pt() > match.candBase->pt();
			}
			else if( match.trackCandBase != 0 )
			{
				return candBase->pt() > match.trackCandBase->pt();
			}
		}
		else if( trackCandBase != 0 )
		{
			if( match.candBase != 0 )
			{
				return trackCandBase->pt() > match.candBase->pt();
			}
			else if( match.trackCandBase != 0 )
			{
				return trackCandBase->pt() > match.trackCandBase->pt();
			}
		}
		//return candBase->pt() > match.candBase->pt();		    	
	}
};

//! Helper structure to order MatchStruct
struct matchesByDescendingPt 
{
	bool operator() (MatchStruct a, MatchStruct b) 
	{     
		bool checked = false;
		if( a.candBase != 0 )
		{
			if( b.candBase != 0 )
			{
				checked = a.candBase->pt() > b.candBase->pt();
			}
			else if( b.trackCandBase != 0 )
			{
				checked = a.candBase->pt() > b.trackCandBase->pt();
			}
		}
		else if( a.trackCandBase != 0 )
		{
			if( b.candBase != 0 )
			{
				checked = a.trackCandBase->pt() > b.candBase->pt();
			}
			else if( b.trackCandBase != 0 )
			{
				checked = a.trackCandBase->pt() > b.trackCandBase->pt();
			}
		}
		//return a.candBase->pt() > b.candBase->pt();
		return checked;
	}
};
#endif
