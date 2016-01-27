#ifndef FWCore_MessageService_NamedDestination_h
#define FWCore_MessageService_NamedDestination_h 1

#include "FWCore/MessageService/interface/ELdestination.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <string>
#include <memory>

class ELdestination;

namespace edm  {
namespace service {       

  class NamedDestination {
  public:
    NamedDestination( std::string const & n, ELdestination* dp ) :
        name_(n)
      , dest_p_(dp)
      {}
    std::string const & name() const {return name_;}
    edm::propagate_const<std::unique_ptr<ELdestination>>& dest_p() {return dest_p_;}
  private:
    std::string name_;
    edm::propagate_const<std::unique_ptr<ELdestination>> dest_p_;
  };
}        // end of namespace service
}       // end of namespace edm

#endif // FWCore_MessageService_NamedDestination_h
