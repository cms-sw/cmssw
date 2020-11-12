#ifndef EventFilter_GEMRawToDigi_GEMRawToDigi_h
#define EventFilter_GEMRawToDigi_GEMRawToDigi_h
/** \class GEMRawToDigi
 *  \author J. Lee, Yechan Kang - UoS
 */
#include <memory>
#include "DataFormats/GEMDigi/interface/AMC13Event.h"

class GEMRawToDigi {
public:
  std::unique_ptr<gem::AMC13Event> convertWordToAMC13Event(const uint64_t* word);
  bool vfatError() const { return vfatError_; }
  bool amcError() const { return amcError_; }

private:
  bool vfatError_;
  bool amcError_;
};
#endif
