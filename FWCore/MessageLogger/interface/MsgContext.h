#ifndef MessageLogger_MsgContext_h 
#define MessageLogger_MsgContext_h 21

#include "FWCore/MessageLogger/interface/ELcontextSupplier.h"

#include <string>

namespace edm
{
  class MsgContext : public edm::ELcontextSupplier
  {
  public:
    MsgContext* clone() const { return new MsgContext(*this); }
    ELstring context() const { return context_; }
    ELstring summaryContext() const { return context(); }
    ELstring fullContext() const { return context(); }

    void setContext(const std::string& c) { context_ = c; }
    void clearContext() { context_.clear(); }

  private:
    std::string context_;
  };

}

#endif
