#ifndef DataFormats_FWLite_LumiHistoryGetter_h
#define DataFormats_FWLite_LumiHistoryGetter_h
// -*- C++ -*-
//
// Package:     DataFormats
// Class  :     LumiHistoryGetter
//
/**\class LumiHistoryGetter LumiHistoryGetter.h src/DataFormats/interface/LumiHistoryGetter.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Wed Feb 10 11:15:16 CST 2010
//

#include "DataFormats/FWLite/interface/LuminosityBlock.h"
#include "DataFormats/FWLite/interface/HistoryGetterBase.h"

namespace fwlite {
    class LumiHistoryGetter : public HistoryGetterBase{
        public:
            LumiHistoryGetter(const LuminosityBlock*);
            virtual ~LumiHistoryGetter();

            // ---------- const member functions ---------------------
            const edm::ProcessHistory& history() const;

        private:
            LumiHistoryGetter(const LumiHistoryGetter&); // stop default

            const LumiHistoryGetter& operator=(const LumiHistoryGetter&); // stop default

            // ---------- member data --------------------------------
            const fwlite::LuminosityBlock* lumi_;
    };

}

#endif
