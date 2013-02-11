#ifndef _SimTrackMatchManager_h_
#define _SimTrackMatchManager_h_

/**\class SimTrackMatchManager

 Description: Matching of SIM and Trigger info for a SimTrack in CSC & GEM

 It's a manager-matcher class, as it uses specialized matching classes to match SimHits, various digis and stubs.

 Original Author:  "Vadim Khotilovich"
 $Id$

*/

#include "BaseMatcher.h"
#include "SimHitMatcher.h"
#include "GEMDigiMatcher.h"
#include "CSCDigiMatcher.h"
//#include "CSCStubMatcher.h"

class SimTrackMatchManager
{
public:
  
  SimTrackMatchManager(const SimTrack* t, const SimVertex* v,
      const edm::ParameterSet* ps, const edm::Event* ev, const edm::EventSetup* es);
  
  ~SimTrackMatchManager();

  SimHitMatcher& simhits() {return simhits_;}
  GEMDigiMatcher& gemDigis() {return gem_digis_;}
  CSCDigiMatcher& cscDigis() {return csc_digis_;}
  //CSCStubMatcher& alcts() {return stubs_;}
  
private:

  SimHitMatcher simhits_;
  GEMDigiMatcher gem_digis_;
  CSCDigiMatcher csc_digis_;
  //CSCStubMatcher stubs_;
};

#endif
