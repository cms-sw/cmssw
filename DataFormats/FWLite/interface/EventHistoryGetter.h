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

#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/HistoryGetterBase.h"

namespace fwlite {
    class EventHistoryGetter : public HistoryGetterBase{
        public:
            EventHistoryGetter(const Event*);
            ~EventHistoryGetter() override;

            // ---------- const member functions ---------------------
            const edm::ProcessHistory& history() const override;

        private:
            EventHistoryGetter(const EventHistoryGetter&) = delete; // stop default

            const EventHistoryGetter& operator=(const EventHistoryGetter&) = delete; // stop default

            // ---------- member data --------------------------------
            const fwlite::Event* event_;
    };

}

#endif
