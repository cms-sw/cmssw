#ifndef LaserAlignment_OrderedLaserHitPair_H
#define LaserAlignment_OrderedLaserHitPair_H


#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

class OrderedLaserHitPair {
public:

  typedef TrackingRecHit OuterHit;
  typedef TrackingRecHit InnerHit;

 

   OrderedLaserHitPair( const InnerHit * ih, const OuterHit * oh)
     : theInnerHit(ih), theOuterHit(oh)  { }



  const InnerHit * inner() const { return theInnerHit; }
  const OuterHit * outer() const { return theOuterHit; } 
private:
  const  InnerHit* theInnerHit;
 const   OuterHit* theOuterHit;
};

#endif

