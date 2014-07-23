#include "DetectorDescription/Core/interface/DDComparator.h"
#include <map>
#include <iostream>

// reason for the ctor: reference initialization at construction.
// FIXME: DDCompareEqual: use pointers instead of references, initialize to 0 and 
// FIXME: do the check in operator() instead of in the ctor

bool DDCompareEqual::operator() (const DDGeoHistory &, const DDPartSelection &) 
{
  return (*this)();
}

bool DDCompareEqual::operator() () 
{

  // don't compare, if history or partsel is empty! (see ctor) 
  bool result(absResult_);
  
  /*
     sIndex_  =  running index in the part-selection-std::vector
     sMax_    =  max. value + 1 of sIndex_
     hIndex_  =  runninig index in the geo-history-std::vector
     hMax_    =  max. value + 1 of hIndex_
     sLp_     =  current LogicalPart (the redir-ptr!) in the part-selection-std::vector
     hLp_     =  current LogicalPart (the redir-ptr!) in the geo-history-std::vector
     sCopyno_ =  current copy-no in the part-selection-std::vector
  */
  //DCOUT('U', "DDCompareEqual: comparing");
    
  while(result && sIndex_ < sMax_) {
    sLp_ = partsel_[sIndex_].lp_;
    sCopyno_ = partsel_[sIndex_].copyno_;
    ddselection_type stype = partsel_[sIndex_].selectionType_; 
    switch (stype) {
   
     case ddanylogp:
        result=nextAnylogp(); 
        break;
   
     case ddanyposp:
        result=nextAnyposp();
        break;
	 
     case ddchildlogp:
        result=nextChildlogp();
	break;
	
     case ddchildposp:
        result=nextChildposp();
	break;
	
     case ddanychild:
        ++sIndex_;
	++hIndex_;
	result=true;
	break;
     
     // ddanynode IS NOT SUPPORTED IN PROTOTYPE SW !!!!
     case ddanynode:
        result=false;
	break;
		
     default:
      result=false;
      //throw DDException("DDCompareEqual: undefined state!");
    }
    ++sIndex_;
  }
  return result;
}


bool DDCompareEqual::nextAnylogp()
{
  register size_t hi = hIndex_;
  while (hi < hMax_) {
    if (sLp_==hist_[hi].logicalPart()) {
      hIndex_ = hi+1;
      return true;
    }
    ++hi;  
  }
  hIndex_ = hi;
  return false;
}


bool DDCompareEqual::nextAnyposp()
{
  bool result(false);
  while (hIndex_ < hMax_) {
    if (sLp_ == hist_[hIndex_].logicalPart() && 
        sCopyno_ == hist_[hIndex_].copyno() ) 
     { result=true;
       ++hIndex_; 
       break; 
     }
    ++hIndex_;
  }    
  return result;
}


bool DDCompareEqual::nextChildlogp()
{
  bool result(false);
  if (hIndex_ < hMax_) {
    if (sLp_ == hist_[hIndex_].logicalPart()) {
      ++hIndex_;
      result=true;
    }
  }
  return result;
}


bool DDCompareEqual::nextChildposp()
{
  bool result(false);
  if (hIndex_ < hMax_) {
    if (sLp_ == hist_[hIndex_].logicalPart() &&
        sCopyno_ == hist_[hIndex_].copyno() ) {
      ++hIndex_;
      result=true;
    }
  }
  return result;
}
