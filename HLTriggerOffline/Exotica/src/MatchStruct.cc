#ifndef HLTRIGGEROFFLINE_EXOTICA_MATCHSTRUCT_CC
#define HLTRIGGEROFFLINE_EXOTICA_MATCHSTRUCT_CC

/** \class MatchStruct
 *  Generate histograms for trigger efficiencies Exotica related
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/EXOTICATriggerValidation
 *
 *  \author  J. Duarte Campderros
 *
 */

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TLorentzVector.h"

#include <vector>

/// MatchStruct helper structure to match gen/reco candidates with
/// HLT trigger objects
struct MatchStruct 
{
	unsigned int objType;
	float pt;
	float eta;
	float phi;
        // Isn't this, like, absolutely dangerous??? 
	const void * thepointer;
  
        /// Default constructor
	MatchStruct():
		objType(0),
		pt(0),
		eta(0),
		phi(0),
		thepointer(0)
	{
	}
  
        /// Constructor from candidate
	MatchStruct(const reco::Candidate * cand, const unsigned int & obj) :
		objType(obj),
		pt(cand->pt()),
		eta(cand->eta()),
		phi(cand->phi()),
		thepointer(cand)

	{
	}
	
        /// Constructor from track
        // FIXME: If finally the track is disappeared, then recover the last code...
	MatchStruct(const reco::Track * cand, const unsigned int & obj) :
		objType(obj),
		pt(cand->pt()),
		eta(cand->eta()),
		phi(cand->phi()),
		thepointer(cand)
	{
	}
	bool operator<(MatchStruct match) 
	{      
		return this->pt < match.pt;
	}
	bool operator>(MatchStruct match) 
	{
		return this->pt > match.pt;		    	
	}
};

/// Helper structure to order MatchStruct
struct matchesByDescendingPt 
{
	bool operator() (MatchStruct a, MatchStruct b) 
	{     
		return a.pt > b.pt;
	}
};
#endif
