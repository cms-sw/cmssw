#ifndef CondFormats_RunInfo_MagneticFieldScaler_h
#define CondFormats_RunInfo_MagneticFieldScaler_h
/** 
 * $Id: HLTScaler.h,v 1.1 2009/02/19 15:58:59 xiezhen Exp $
 *
 ************************************************************/
#include <vector>
class MagneticFieldScaler{
 public:
    MagneticFieldScaler();
    //maximum length of 321 (the number of scalable volumes)
    std::vector<int> scalingvolumes;
    std::vector<double> scalingfactors;
}; 
#endif 
