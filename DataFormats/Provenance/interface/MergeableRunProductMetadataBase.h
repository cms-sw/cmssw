#include <string>

namespace edm {

  class MergeableRunProductMetadataBase {
  public:
    virtual bool knownImproperlyMerged(std::string const& processThatCreatedProduct) const = 0;
  };
}
