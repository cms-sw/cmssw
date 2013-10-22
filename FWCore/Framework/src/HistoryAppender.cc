#include "FWCore/Framework/interface/HistoryAppender.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <string>
#include <cassert>

static const edm::ProcessHistory s_emptyHistory;

namespace edm {

  HistoryAppender::HistoryAppender()
  {
  }

  boost::shared_ptr<ProcessHistory const>
  HistoryAppender::appendToProcessHistory(ProcessHistoryID const& inputPHID,
                                          ProcessHistory const* iInputProcessHistory,
                                          ProcessConfiguration const& pc)  {
    assert((iInputProcessHistory) == nullptr or (inputPHID == iInputProcessHistory->id()));
    if (m_cachedHistory.get() != nullptr and inputPHID==m_cachedInputPHID) {
      return m_cachedHistory;
    }

    ProcessHistory const* inputProcessHistory = iInputProcessHistory? iInputProcessHistory : &s_emptyHistory;

    if (inputPHID.isValid()) {
      if (iInputProcessHistory == nullptr) {
        throw Exception(errors::LogicError)
          << "HistoryAppender::appendToProcessHistory\n"
          << "Input ProcessHistory has valid ID but is nullptr\n"
          << "Contact a Framework developer\n";
      }
    }

    boost::shared_ptr<ProcessHistory> newProcessHistory(new ProcessHistory);
    *newProcessHistory = *inputProcessHistory;
    checkProcessHistory(*newProcessHistory, pc);
    newProcessHistory->push_back(pc);
    //force it to create the ID
    newProcessHistory->setProcessHistoryID();
    m_cachedInputPHID =inputPHID;
    m_cachedHistory = newProcessHistory;
    return m_cachedHistory;
  }

  void
  HistoryAppender::checkProcessHistory(ProcessHistory const& ph,
                                       ProcessConfiguration const& pc) const {
    std::string const& processName = pc.processName();
    for (auto const& process : ph) {
      if (processName == process.processName()) {
        throw edm::Exception(errors::Configuration, "Duplicate Process.")
          << "The process name " << processName << " was already in the ProcessHistory.\n"
          << "Please modify the configuration file to use a distinct process name.\n";
      }
    }
  }
}
