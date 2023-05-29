#include "FWCore/Framework/interface/resolveMaker.h"

namespace edm::detail {
  void annotateResolverMakerExceptionAndRethrow(cms::Exception& except,
                                                std::string const& modtype,
                                                ModuleTypeResolverBase const* resolver) {
    if (not resolver) {
      throw except;
    }
    //if needed, create list of alternative types that were tried
    std::string alternativeTypes;
    auto index = resolver->kInitialIndex;
    auto newType = modtype;
    int tries = 0;
    do {
      ++tries;
      if (not alternativeTypes.empty()) {
        alternativeTypes.append(", ");
      }
      auto [ttype, tindex] = resolver->resolveType(std::move(newType), index);
      newType = std::move(ttype);
      index = tindex;
      alternativeTypes.append(newType);
    } while (index != resolver->kLastIndex);
    if (tries == 1 and alternativeTypes == modtype) {
      throw except;
    }
    alternativeTypes.insert(0, "These alternative types were tried: ");
    except.addAdditionalInfo(alternativeTypes);
    throw except;
  }
}  // namespace edm::detail
