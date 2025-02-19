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
// $Id: HistoryGetterBase.h,v 1.1 2010/02/11 17:21:38 ewv Exp $
//
#if !defined(__CINT__) && !defined(__MAKECINT__)

#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"

namespace fwlite {
    class HistoryGetterBase {
        public:
            HistoryGetterBase();
            virtual ~HistoryGetterBase();

            // ---------- const member functions ---------------------
            virtual const edm::ProcessHistory& history() const = 0;

        private:
            HistoryGetterBase(const HistoryGetterBase&); // stop default

            const HistoryGetterBase& operator=(const HistoryGetterBase&); // stop default

            // ---------- member data --------------------------------
    };
}

#endif /*__CINT__ */

#endif
