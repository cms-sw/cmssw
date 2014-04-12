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
#if !defined(__CINT__) && !defined(__MAKECINT__)

#include "DataFormats/FWLite/interface/Run.h"
#include "DataFormats/FWLite/interface/HistoryGetterBase.h"

namespace fwlite {
    class RunHistoryGetter : public HistoryGetterBase{
        public:
            RunHistoryGetter(const Run*);
            virtual ~RunHistoryGetter();

            // ---------- const member functions ---------------------
            const edm::ProcessHistory& history() const;

        private:
            RunHistoryGetter(const RunHistoryGetter&); // stop default

            const RunHistoryGetter& operator=(const RunHistoryGetter&); // stop default

            // ---------- member data --------------------------------
            const fwlite::Run* run_;
    };

}

#endif /*__CINT__ */
#endif
