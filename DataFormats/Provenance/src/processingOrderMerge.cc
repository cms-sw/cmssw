#include "DataFormats/Provenance/interface/processingOrderMerge.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>

namespace edm {
  namespace {
    [[noreturn]] void throwIncompatibleOrdering(std::string const& msg) {
      throw cms::Exception("IncompatibleProcessHistoryOrdering")
          << "Incompatible process history ordering during merge: " << msg << "\n";
    }

    template <typename Iter>
    Iter foundLaterIn(Iter itBegin, Iter it, Iter itEnd, std::string const& name) {
      auto itFind = std::find(itBegin, itEnd, name);
      // Sanity check, it should be impossible for this to happen
      if (itFind < it) {
        throwIncompatibleOrdering("process " + name + " found earlier in other list");
      }
      return itFind;
    }
  }  // namespace

  void processingOrderMerge(ProcessHistory const& iHistory, std::vector<std::string>& processNames) {
    if (processNames.empty()) {
      for (auto it = iHistory.rbegin(); it != iHistory.rend(); ++it) {
        processNames.push_back(it->processName());
      }
    } else {
      std::vector<std::string> fromHistory;
      fromHistory.reserve(iHistory.size());
      for (auto it = iHistory.rbegin(); it != iHistory.rend(); ++it) {
        fromHistory.push_back(it->processName());
      }
      processingOrderMerge(fromHistory, processNames);
    }
  }

  void processingOrderMerge(std::vector<std::string> const& iHistory, std::vector<std::string>& processNames) {
    if (processNames.empty()) {
      processNames = iHistory;
    } else {
      std::vector<std::string> tempNames;
      tempNames.reserve(processNames.size() > iHistory.size() ? processNames.size() : iHistory.size());
      auto itNew = iHistory.begin();
      auto itNewEnd = iHistory.end();
      auto itOld = processNames.begin();
      auto itOldEnd = processNames.end();
      while (itNew != itNewEnd && itOld != itOldEnd) {
        if (*itNew == *itOld) {
          tempNames.push_back(*itOld);
          ++itNew;
          ++itOld;
        } else {
          //see if we can find it in the old list
          auto itFindOld = foundLaterIn(processNames.begin(), itOld, itOldEnd, *itNew);
          auto itFindNew = foundLaterIn(iHistory.begin(), itNew, itNewEnd, *itOld);
          if (itFindOld != itOldEnd) {
            if (itFindNew != itNewEnd) {
              throwIncompatibleOrdering("order of processes " + *itNew + " and " + *itOld +
                                        " is not the same in all ProcessHistories");
            }
            tempNames.push_back(*itOld);
            ++itOld;
          } else {
            if (itFindNew == itNewEnd) {
              throwIncompatibleOrdering("order of processes " + *itOld + " and " + *itNew + " is ambiguous");
            }
            tempNames.push_back(*itNew);
            ++itNew;
          }
        }
      }
      //copy over any remaining old names
      while (itOld != itOldEnd) {
        tempNames.push_back(*itOld);
        ++itOld;
      }
      while (itNew != itNewEnd) {
        tempNames.push_back(*itNew);
        ++itNew;
      }
      processNames.swap(tempNames);
    }
  }
}  // namespace edm
