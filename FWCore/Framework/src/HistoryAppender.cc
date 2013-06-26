#include "FWCore/Framework/interface/HistoryAppender.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <string>

namespace edm {

  HistoryAppender::HistoryAppender() :
    previous_(historyMap_.end()) {
  }

  CachedHistory const&
  HistoryAppender::appendToProcessHistory(ProcessHistoryID const& inputPHID,
                                          ProcessConfiguration const& pc) {

    if (previous_ != historyMap_.end() && inputPHID == previous_->first) return previous_->second;

    HistoryMap::iterator iter = historyMap_.find(inputPHID);
    if (iter != historyMap_.end()) return iter->second;

    ProcessHistoryRegistry* registry = ProcessHistoryRegistry::instance();

    ProcessHistory const* inputProcessHistory = &emptyHistory_;

    if (inputPHID.isValid()) {
      inputProcessHistory = registry->getMapped(inputPHID);
      if (inputProcessHistory == 0) {
        throw Exception(errors::LogicError)
          << "HistoryAppender::appendToProcessHistory\n"
          << "Input ProcessHistory not found in registry\n"
          << "Contact a Framework developer\n";
      }
    }

    ProcessHistory newProcessHistory;
    newProcessHistory = *inputProcessHistory;
    checkProcessHistory(newProcessHistory, pc);
    newProcessHistory.push_back(pc);
    registry->insertMapped(newProcessHistory);
    ProcessHistoryID newProcessHistoryID = newProcessHistory.id();
    CachedHistory newValue(inputProcessHistory,
                           registry->getMapped(newProcessHistoryID),
                           newProcessHistoryID);
    std::pair<ProcessHistoryID, CachedHistory> newEntry(inputPHID, newValue);
    std::pair<HistoryMap::iterator, bool> result = historyMap_.insert(newEntry);
    previous_ = result.first;
    return result.first->second;
  }

  void
  HistoryAppender::checkProcessHistory(ProcessHistory const& ph,
                                       ProcessConfiguration const& pc) const {
    std::string const& processName = pc.processName();
    for (ProcessHistory::const_iterator it = ph.begin(), itEnd = ph.end(); it != itEnd; ++it) {
      if (processName == it->processName()) {
        throw edm::Exception(errors::Configuration, "Duplicate Process.")
          << "The process name " << processName << " was already in the ProcessHistory.\n"
          << "Please modify the configuration file to use a distinct process name.\n";
      }
    }
  }
}
