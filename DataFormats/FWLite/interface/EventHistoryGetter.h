#ifndef DataFormats_FWLite_EventHistoryGetter_h
#define DataFormats_FWLite_EventHistoryGetter_h
// -*- C++ -*-
//
// Package:     DataFormats
// Class  :     EventHistoryGetter
//
/**\class EventHistoryGetter EventHistoryGetter.h src/DataFormats/interface/EventHistoryGetter.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Wed Feb 10 11:15:16 CST 2010
//
#if !defined(__CINT__) && !defined(__MAKECINT__)

#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/HistoryGetterBase.h"

namespace fwlite {
    class EventHistoryGetter : public HistoryGetterBase{
        public:
            EventHistoryGetter(const Event*);
            virtual ~EventHistoryGetter();

            // ---------- const member functions ---------------------
            const edm::ProcessHistory& history() const;

        private:
            EventHistoryGetter(const EventHistoryGetter&); // stop default

            const EventHistoryGetter& operator=(const EventHistoryGetter&); // stop default

            // ---------- member data --------------------------------
            const fwlite::Event* event_;
    };

}

#endif /*__CINT__ */
#endif
