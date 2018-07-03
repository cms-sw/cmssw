#ifndef DETECTOR_DESCRIPTION_CORE_DD_COMPARATOR_H
#define DETECTOR_DESCRIPTION_CORE_DD_COMPARATOR_H

#include <vector>

#include "DetectorDescription/Core/interface/DDExpandedNode.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDPartSelection.h"


//! compares a given geometrical-history whether it corresponds to the given part-selector
/**
  This is a function-object.
*/
class DDCompareEqual
{
public:
  
 DDCompareEqual(const DDGeoHistory & h, const DDPartSelection & s)
   : hist_(h), 
    partsel_(s), 
    hMax_(h.size()), 
    hIndex_(0), 
    sMax_(s.size()), 
    sIndex_(0), 
    sLp_(), 
    sCopyno_(0), 
    absResult_(hMax_>0 && sMax_>0 ) 
    { 
      // it makes only sense to compare if both std::vectors have at least one entry each.
    }

  DDCompareEqual() = delete;  

  bool operator() (const DDGeoHistory &, const DDPartSelection &);
  bool operator() ();

protected:
  inline bool nextAnylogp();
  inline bool nextAnyposp();
  inline bool nextChildlogp();
  inline bool nextChildposp();
  
private:
  const DDGeoHistory & hist_;
  const DDPartSelection & partsel_;
  DDGeoHistory::size_type const hMax_;
  DDGeoHistory::size_type hIndex_;
  DDPartSelection::size_type const sMax_;
  DDPartSelection::size_type sIndex_;
  DDLogicalPart sLp_;
  int sCopyno_;
  bool absResult_;
};

#endif
