#ifndef DataFormats_FWLite_RunHistoryGetter_h
#define DataFormats_FWLite_RunHistoryGetter_h
// -*- C++ -*-
//
// Package:     DataFormats
// Class  :     RunHistoryGetter
//
/**\class RunHistoryGetter RunHistoryGetter.h src/DataFormats/interface/RunHistoryGetter.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Wed Feb 10 11:15:16 CST 2010
//

#include "DataFormats/FWLite/interface/Run.h"
#include "DataFormats/FWLite/interface/HistoryGetterBase.h"

namespace fwlite {
    class RunHistoryGetter : public HistoryGetterBase{
        public:
            RunHistoryGetter(const Run*);
            ~RunHistoryGetter() override;

            // ---------- const member functions ---------------------
            const edm::ProcessHistory& history() const override;

        private:
            RunHistoryGetter(const RunHistoryGetter&) = delete; // stop default

            const RunHistoryGetter& operator=(const RunHistoryGetter&) = delete; // stop default

            // ---------- member data --------------------------------
            const fwlite::Run* run_;
    };

}

#endif
