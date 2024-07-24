#ifndef L1Trigger_DTTriggerPhase2_ShowerBuffer_h
#define L1Trigger_DTTriggerPhase2_ShowerBuffer_h
#include <iostream>
#include <memory>

#include "L1Trigger/DTTriggerPhase2/interface/DTprimitive.h"

class ShowerBuffer {
public:
  ShowerBuffer();
  virtual ~ShowerBuffer() {}

  // setter methods
  void addHit(DTPrimitive &prim);
  void rawId(int rawId) { rawId_ = rawId;}
 
  // Get nHits
  int getNhits() { return nprimitives_; }
  bool isFlagged(){return shower_flag_;}
  DTPrimitivePtrs getHits() { return prim_; }

  // Other methods
  int getRawId(){return rawId_;}
  void flag(){shower_flag_ = true;}

private:
  //------------------------------------------------------------------
  //---  ShowerBuffer's data
  //------------------------------------------------------------------
  /*
      Primitives that make up the path. The 0th position holds the channel ID of 
      the lower layer. The order is critical. 
  */
  DTPrimitivePtrs prim_;  
  short nprimitives_;
  int rawId_;
  bool shower_flag_;
};

typedef std::vector<ShowerBuffer> ShowerBuffers;
typedef std::shared_ptr<ShowerBuffer> ShowerBufferPtr;
typedef std::vector<ShowerBufferPtr> ShowerBufferPtrs;

#endif
