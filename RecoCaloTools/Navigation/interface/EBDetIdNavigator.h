#ifndef RECOCALOTOOLS_NAVIGATION_EBDETIDNAVIGATOR_H
#define RECOCALOTOOLS_NAVIGATION_EBDETIDNAVIGATOR_H 1

#include "DataFormats/EcalDetId/interface/EBDetId.h"

/** \class EBDetIdNavigator
  *  
  * $Date: 2006/01/17 22:50:43 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class EBDetIdNavigator {
public:
  /// create a new navigator
  explicit EBDetIdNavigator(const EBDetId& startingPoint);

  /// move the navigator back to the starting point
  void home();

  /// get the current position
  EBDetId pos() const { return currentPoint_; }

  /// get the current position
  EBDetId operator*() const { return currentPoint_; }

  /// move the nagivator to larger ieta (more positive z) (stops at end of barrel and returns null)
  EBDetId incrementIeta();

  /// move the nagivator to smaller ieta (more negative z) (stops at end of barrel and returns null)
  EBDetId decrementIeta();

  /// move the nagivator to larger iphi (wraps around the barrel) 
  EBDetId incrementIphi();

  /// move the nagivator to smaller iphi (wraps around the barrel)
  EBDetId decrementIphi();

  /// move the navigator north (increment iphi)
  EBDetId north() { return incrementIphi(); }

  /// move the navigator south (decrement iphi)
  EBDetId south() { return decrementIphi(); }

  /// move the navigator east (negative ieta)
  EBDetId east() { return decrementIeta(); }

  /// move the navigator west (positive ieta)
  EBDetId west() { return incrementIeta(); }
 
private:
  EBDetId startingPoint_, currentPoint_;
};

#endif
