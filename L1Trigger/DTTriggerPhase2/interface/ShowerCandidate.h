#ifndef L1Trigger_DTTriggerPhase2_ShowerCandidate_h
#define L1Trigger_DTTriggerPhase2_ShowerCandidate_h
#include <iostream>
#include <memory>

#include "L1Trigger/DTTriggerPhase2/interface/DTprimitive.h"
#include "L1Trigger/DTTriggerPhase2/interface/ShowerBuffer.h"

class ShowerCandidate {
public:
  ShowerCandidate();
  ShowerCandidate(ShowerBuffer& buff);
  ShowerCandidate(ShowerBufferPtr& buffPtr);
  virtual ~ShowerCandidate() {}

  // setter methods
  void rawId(int rawId) { rawId_ = rawId;}
  void setAvgTime();
  void setAvgPos();
 
  // Get nHits
  int getNhits() { return nhits_; }

  // Other methods
  int getRawId(){return rawId_;}
  float getAvgTime() {return avgTime_;}
  float getAvgPos() {return avgPos_;};


private:
  //------------------------------------------------------------------
  //---  ShowerCandidate's data
  //------------------------------------------------------------------
  DTPrimitivePtrs hits_;  
  int nhits_;
  int rawId_;

  float avgTime_;
  float avgPos_;
};

typedef std::vector<ShowerCandidate> ShowerCandidates;
typedef std::shared_ptr<ShowerCandidate> ShowerCandidatePtr;
typedef std::vector<ShowerCandidatePtr> ShowerCandidatePtrs;

#endif
