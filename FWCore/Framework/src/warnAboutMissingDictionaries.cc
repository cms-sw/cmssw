#include "FWCore/Framework/interface/warnAboutMissingDictionaries.h"

#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Reflection/interface/DictionaryTools.h"

#include "TClassEdit.h"

#include "oneapi/tbb/concurrent_unordered_set.h"

#include <set>
#include <string>

namespace edm {
  void warnAboutMissingDictionaries(ProductDescription const& prod) {
    if (prod.transient()) {
      return;
    }
    // Scan each product type only once.
    static oneapi::tbb::concurrent_unordered_set<std::string> alreadyCheckedTypes;
    if (!alreadyCheckedTypes.emplace(prod.className()).second) {
      return;
    }
    std::set<std::string> missingDictionaries;
    findClassesWithoutDictionaries(missingDictionaries, prod.unwrappedType());
    if (missingDictionaries.empty()) {
      return;
    }
    edm::LogWarning log("MissingDictionaries");
    log << "No data dictionary found for the following classes, needed by product\n'" << prod.branchName()
        << "' of type '" << prod.className() << "':\n\n";
    for (auto const& name : missingDictionaries) {
      // Spelled as in classes_def.xml.
      log << "  " << TClassEdit::InsertStd(name.c_str()) << "\n";
    }
    log << "\nROOT falls back to the much slower interpreter (cling) or streamer emulation\n"
        << "for them. Add '<class name=\"...\"/>' for each to the classes_def.xml of the\n"
        << "package that defines it (for a container, of its elements), with the header\n"
        << "in classes.h, or drop the product if it is not needed.";
  }
}  // namespace edm
