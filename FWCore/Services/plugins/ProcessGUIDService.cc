#include "FWCore/Utilities/interface/Guid.h"
#include "FWCore/Utilities/interface/ProcessGUID.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

namespace edm::service {
  class ProcessGUIDService : public ProcessGUID {
  public:
    ProcessGUIDService() = default;

    std::string binary() const override { return guid_.toBinary(); }

    std::string string() const override { return guid_.toString(); }

  private:
    Guid guid_;
  };

  bool isProcessWideService(ProcessGUIDService const *) { return true; }
}  // namespace edm::service

using edm::service::ProcessGUIDService;
using ProcessGUIDServiceMaker = edm::serviceregistry::NoArgsMaker<edm::ProcessGUID, ProcessGUIDService>;
DEFINE_FWK_SERVICE_MAKER(ProcessGUIDService, ProcessGUIDServiceMaker);
