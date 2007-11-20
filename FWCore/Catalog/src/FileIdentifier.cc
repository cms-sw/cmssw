#include "FWCore/Catalog/interface/FileIdentifier.h"
#include "POOLCore/Guid.h"

namespace edm {
  std::string
  createFileIdentifier() {
    pool::Guid guid;
    pool::Guid::create(guid);
    return guid.toString();
  }
}
