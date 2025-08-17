#include "DataFormats/Provenance/interface/processingOrderMerge.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {
  namespace {
    [[noreturn]] void throwIncompatibleOrdering(std::string const& msg) {
      throw cms::Exception("IncompatibleProcessHistoryOrdering")
          << "Incompatible process history ordering during merge: " << msg << "\n";
    }

    template <typename Iter>
    Iter foundLaterIn(Iter itBegin, Iter it, Iter itEnd, std::string const& name) {
      auto itFind = std::find(itBegin, itEnd, name);
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
      ;
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
              throwIncompatibleOrdering("process " + *itNew + " and " + *itOld + " are out of order");
            }
            //found it, copy over everything up to and including it
            while (itOld != itFindOld) {
              tempNames.push_back(*itOld);
              ++itOld;
            }
            tempNames.push_back(*itOld);
            ++itOld;
            ++itNew;
          } else {
            if (itFindNew == itNewEnd) {
              throwIncompatibleOrdering("process " + *itOld + " and " + *itNew + " are independent");
            }
            while (itNew != itFindNew) {
              tempNames.push_back(*itNew);
              ++itNew;
            }
            //not found, add the new one
            tempNames.push_back(*itNew);
            ++itNew;
            ++itOld;
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
