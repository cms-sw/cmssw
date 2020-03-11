#ifndef DataFormats_FWLite_HistoryGetterBase_h
#define DataFormats_FWLite_HistoryGetterBase_h
// -*- C++ -*-
//
// Package:     DataFormats
// Class  :     HistoryGetterBase
//
/**\class HistoryGetterBase HistoryGetterBase.h src/DataFormats/interface/HistoryGetterBase.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Wed Feb 10 11:15:16 CST 2010
//

#include "DataFormats/Provenance/interface/ProcessHistory.h"

namespace fwlite {
  class HistoryGetterBase {
  public:
    HistoryGetterBase();
    virtual ~HistoryGetterBase();

    // ---------- const member functions ---------------------
    virtual const edm::ProcessHistory& history() const = 0;

  private:
    HistoryGetterBase(const HistoryGetterBase&) = delete;  // stop default

    const HistoryGetterBase& operator=(const HistoryGetterBase&) = delete;  // stop default

    // ---------- member data --------------------------------
  };
}  // namespace fwlite

#endif
