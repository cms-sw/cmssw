#ifndef LaserAlignment_OrderedLaserHitPair_H
#define LaserAlignment_OrderedLaserHitPair_H

/** \class OrderedLaserHitPair
 *  pair of ordered laser hits 
 *
 *  $Date: 2007/05/10 12:00:32 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */

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

