#ifndef DataFormats_Provenance_MergeableRunProductMetadataBase_h
#define DataFormats_Provenance_MergeableRunProductMetadataBase_h
#include <string>

namespace edm {

  class MergeableRunProductMetadataBase {
  public:
    virtual ~MergeableRunProductMetadataBase();

    virtual bool knownImproperlyMerged(std::string const& processThatCreatedProduct) const = 0;
  };
}  // namespace edm
#endif
