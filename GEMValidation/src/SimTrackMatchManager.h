#ifndef _SimTrackMatchManager_h_
#define _SimTrackMatchManager_h_

/**\class SimTrackMatchManager

 Description: Matching of SIM and Trigger info for a SimTrack in CSC & GEM

 It's a manager-matcher class, as it uses specialized matching classes to match SimHits, various digis and stubs.

 Original Author:  "Vadim Khotilovich"
 $Id: SimTrackMatchManager.h,v 1.1 2013/02/11 07:33:07 khotilov Exp $

*/

#include "BaseMatcher.h"
#include "SimHitMatcher.h"
#include "GEMDigiMatcher.h"
#include "CSCDigiMatcher.h"
#include "CSCStubMatcher.h"

class SimTrackMatchManager
{
public:
  
  SimTrackMatchManager(const SimTrack* t, const SimVertex* v,
      const edm::ParameterSet* ps, const edm::Event* ev, const edm::EventSetup* es);
  
  ~SimTrackMatchManager();

  const SimHitMatcher& simhits() const {return simhits_;}
  const GEMDigiMatcher& gemDigis() const {return gem_digis_;}
  const CSCDigiMatcher& cscDigis() const {return csc_digis_;}
  const CSCStubMatcher& cscStubs() const {return stubs_;}
  
private:

  SimHitMatcher simhits_;
  GEMDigiMatcher gem_digis_;
  CSCDigiMatcher csc_digis_;
  CSCStubMatcher stubs_;
};

#endif
