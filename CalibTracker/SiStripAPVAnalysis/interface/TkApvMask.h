#ifndef TkAPVMask_H
#define TkAPVMask_H

#include "CalibTracker/SiStripAPVAnalysis/interface/ApvAnalysis.h"
#include <vector>
/**
 * The abstract class for dead/noisy/etc strips masking.
 */
class TkApvMask {  
  
 public:
  
  virtual ~TkApvMask() {}

  enum StripMaskType{ok=0,dead=1,noisy=2};
  
  typedef std::vector<StripMaskType> MaskType;
  
  virtual void setMask(const MaskType& in) = 0 ;
  virtual MaskType mask() = 0 ;
  
  virtual void calculateMask(const ApvAnalysis::PedestalType& ) = 0;
  
};

#endif
