#ifndef LaserAlignment_OrderedLaserHitPairs_H
#define LaserAlignment_OrderedLaserHitPairs_H

/** \class OrderedLaserHitPairs
 *  ordered pairs of laser hits; used for seedgenerator
 *
 *  $Date: 2007/05/10 12:00:32 $
 *  $Revision: 1.2 $
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
