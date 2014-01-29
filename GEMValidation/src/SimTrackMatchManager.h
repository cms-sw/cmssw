#ifndef GEMValidation_SimTrackMatchManager_h
#define GEMValidation_SimTrackMatchManager_h

/**\class SimTrackMatchManager

 Description: Matching of SIM and Trigger info for a SimTrack in CSC & GEM

 It's a manager-matcher class, as it uses specialized matching classes to match SimHits, various digis and stubs.

 Original Author:  "Vadim Khotilovich"
*/

#include "GEMCode/GEMValidation/src/BaseMatcher.h"
#include "GEMCode/GEMValidation/src/SimHitMatcher.h"
#include "GEMCode/GEMValidation/src/GEMDigiMatcher.h"
#include "GEMCode/GEMValidation/src/CSCDigiMatcher.h"
#include "GEMCode/GEMValidation/src/CSCStubMatcher.h"
#include "GEMCode/GEMValidation/src/GEMRecHitMatcher.h"

class SimTrackMatchManager
{
public:
  
  SimTrackMatchManager(const SimTrack& t, const SimVertex& v,
      const edm::ParameterSet& ps, const edm::Event& ev, const edm::EventSetup& es);
  
  ~SimTrackMatchManager();

  const SimHitMatcher& simhits() const {return simhits_;}
  const GEMDigiMatcher& gemDigis() const {return gem_digis_;}
  const CSCDigiMatcher& cscDigis() const {return csc_digis_;}
  const CSCStubMatcher& cscStubs() const {return stubs_;}
  const GEMRecHitMatcher& gemRecHits() const {return gem_rechits_;}
  
private:

  SimHitMatcher simhits_;
  GEMDigiMatcher gem_digis_;
  CSCDigiMatcher csc_digis_;
  CSCStubMatcher stubs_;
  GEMRecHitMatcher gem_rechits_;
};

#endif
