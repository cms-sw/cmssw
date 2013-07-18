#ifndef LaserAlignment_OrderedLaserHitPairs_H
#define LaserAlignment_OrderedLaserHitPairs_H

/** \class OrderedLaserHitPairs
 *  ordered pairs of laser hits; used for seedgenerator
 *
 *  $Date: Thu May 10 13:53:49 CEST 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/interface/OrderedLaserHitPair.h"
#include <vector>

class OrderedLaserHitPairs : public std::vector<OrderedLaserHitPair> { 
public:

  virtual ~OrderedLaserHitPairs(){}

  virtual unsigned int size() const { return std::vector<OrderedLaserHitPair>::size(); }

  virtual const OrderedLaserHitPair& operator[](unsigned int i) const { 
    return std::vector<OrderedLaserHitPair>::operator[](i); 
  }

};
#endif
