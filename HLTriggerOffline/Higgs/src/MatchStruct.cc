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

#include<vector>

// Matching structure: helper structure to match gen/reco candidates with
// hlt trigger objects
struct MatchStruct 
{
	unsigned int objType;
	const reco::Candidate * candBase;
	MatchStruct() 
	{
		candBase   = 0;
	}
	MatchStruct(const reco::Candidate * cand, const unsigned int & obj) 
	{
		candBase = cand;
		objType = obj;
	}
	bool operator<(MatchStruct match) 
	{      
		return candBase->pt() < match.candBase->pt();
	}
	bool operator>(MatchStruct match) 
	{
		return candBase->pt() > match.candBase->pt();		    	
	}
};

//! Helper structure to order MatchStruct
struct matchesByDescendingPt 
{
	bool operator() (MatchStruct a, MatchStruct b) 
	{     
		return a.candBase->pt() > b.candBase->pt();
	}
};
#endif
