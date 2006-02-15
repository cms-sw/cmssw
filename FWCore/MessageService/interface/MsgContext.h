#ifndef FWCore_MessageService_MsgContext_h 
#define FWCore_MessageService_MsgContext_h 

#include "FWCore/MessageService/interface/ELcontextSupplier.h"

#include <string>

namespace edm {
namespace service {       

  class MsgContext : public ELcontextSupplier
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

}        // end of namespace service
}       // end of namespace edm

#endif // FWCore_MessageService_MsgContext_h
