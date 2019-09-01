#ifndef Cond_IOVProvenance_h
#define Cond_IOVProvenance_h

#include "CondFormats/Serialization/interface/Serializable.h"

namespace cond {

  class IOVProvenance {
  public:
    IOVProvenance() {}
    virtual ~IOVProvenance() {}
    virtual IOVProvenance* clone() const { return new IOVProvenance(*this); }

  private:
    COND_SERIALIZABLE;
  };

}  // namespace cond

#endif
