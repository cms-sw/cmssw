#ifndef Cond_IOVDescription_h
#define Cond_IOVDescription_h

#include "CondFormats/Serialization/interface/Serializable.h"

namespace cond {

  class IOVDescription {
  public:
    IOVDescription() {}
    virtual ~IOVDescription() {}
    virtual IOVDescription* clone() const { return new IOVDescription(*this); }

  private:
    COND_SERIALIZABLE;
  };

}  // namespace cond

#endif
