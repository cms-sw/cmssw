#include "DataFormats/Provenance/interface/History.h"

namespace edm
{
  
  History::size_type
  History::size() const
  {
    return eventSelections_.size();
  }

  void 
  History::addEntry(EventSelectionID const& eventSelection)
  {
    eventSelections_.push_back(eventSelection);
  }

  void
  History::setProcessHistoryID(ProcessHistoryID const& pid) {
    processHistoryID_ = pid;
  }


  EventSelectionID const&
  History::getEventSelectionID(History::size_type i) const
  {
    return eventSelections_[i];
  }

  EventSelectionIDVector const&
  History::eventSelectionIDs() const
  {
    return eventSelections_;
  }

  ProcessHistoryID const&
  History::processHistoryID() const
  {
    return processHistoryID_;
  }

}
