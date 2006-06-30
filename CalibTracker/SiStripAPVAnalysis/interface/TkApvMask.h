#ifndef TkAPVMask_H
#define TkAPVMask_H

#include "CalibTracker/SiStripAPVAnalysis/interface/ApvAnalysis.h"
#include <vector>
/**
 * The abstract class for dead/noisy/etc strips masking.
 */
class TkApvMask {  
  
 public:

  enum StripMaskType{ok=1,dead=2,noisy=3};
  
  typedef std::vector<StripMaskType> MaskType;
  
  virtual void setMask(MaskType in) = 0 ;
  virtual MaskType mask() = 0 ;
  
  virtual void calculateMask(ApvAnalysis::PedestalType ) = 0;
  
};

#endif
