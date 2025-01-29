/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "DroppedDataProductResolver.h"
#include "FWCore/Framework/interface/ProductProvenanceRetriever.h"

namespace edm {

  DroppedDataProductResolver::Resolution DroppedDataProductResolver::resolveProduct_(
      Principal const& principal,
      bool skipCurrentProcess,
      SharedResourcesAcquirer* sra,
      ModuleCallingContext const* mcc) const {
    return Resolution(nullptr);
  }
  void DroppedDataProductResolver::prefetchAsync_(WaitingTaskHolder waitTask,
                                                  Principal const& principal,
                                                  bool skipCurrentProcess,
                                                  ServiceToken const& token,
                                                  SharedResourcesAcquirer* sra,
                                                  ModuleCallingContext const* mcc) const noexcept {}

  void DroppedDataProductResolver::retrieveAndMerge_(
      Principal const& principal, MergeableRunProductMetadata const* mergeableRunProductMetadata) const {}

  void DroppedDataProductResolver::setProductProvenanceRetriever_(ProductProvenanceRetriever const* provRetriever) {
    m_provenance.setStore(provRetriever);
  }

}  // namespace edm
