#ifndef LaserAlignment_OrderedLaserHitPair_H
#define LaserAlignment_OrderedLaserHitPair_H

/** \class OrderedLaserHitPair
 *  pair of ordered laser hits 
 *
 *  $Date: Thu May 10 13:54:44 CEST 2007 $
 *  $Revision: 1.1 $
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

