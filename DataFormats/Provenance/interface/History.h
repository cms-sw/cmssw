#ifndef DataFormats_Provenance_History_h
#define DataFormats_Provenance_History_h

//----------------------------------------------------------------------
//
// Class History represents the processing history of a single Event.
// It includes ordered sequences of elements, each of which contains
// information about a specific 'process' through which the Event has
// passed, with earlier processes at the beginning of the sequence.
//
//
//----------------------------------------------------------------------

#include <vector>
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

namespace edm
{
  class History
  {
  public:
    typedef std::size_t size_type;
    
    // Compiler-generated default c'tor, copy c'tor, assignment and
    // d'tor are all correct.


    // Return the number of 'processing steps' recorded in this
    // History.
    size_type size() const;
    
    // Add the given entry to this History. When a new data member is
    // added to the History class, this function should be modified to
    // take an instance of the type of the new data member.
    void addEntry(EventSelectionID const& eventSelection);
    
    // Set the process History ID.
    void setProcessHistoryID(ProcessHistoryID const& pid);

    EventSelectionID const& getEventSelectionID(size_type i) const;

    EventSelectionIDVector const& eventSelectionIDs() const;
    
    ProcessHistoryID const& processHistoryID() const;
  private:
    
    // Note: We could, instead, define a struct that contains the
    // appropriate information for each history entry, and then contain
    // only one data member: a vector of this struct. This might make
    // iteration more convenient. But it would seem to complicate
    // persistence. The current plan is to have parallel vectors, one
    // for each type of item stored as data.
    EventSelectionIDVector eventSelections_;

    ProcessHistoryID processHistoryID_;
  };

}

#endif
