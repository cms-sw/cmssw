#ifndef DataFormats_FWLite_RunFactory_h
#define DataFormats_FWLite_RunFactory_h
// -*- C++ -*-
//
// Package:     DataFormats/FWLite
// Class  :     RunFactory
//
/**\class RunFactory RunFactory.h src/DataFormats/interface/RunFactory.h

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

#include <boost/shared_ptr.hpp>
#include "DataFormats/FWLite/interface/Run.h"
#include "FWCore/FWLite/interface/BranchMapReader.h"

namespace fwlite {
    class RunFactory {
        public:
            RunFactory();
            virtual ~RunFactory();

            // ---------- const member functions ---------------------
            boost::shared_ptr<fwlite::Run> makeRun(boost::shared_ptr<BranchMapReader> branchMap) const;

        private:
            RunFactory(const RunFactory&); // stop default

            const RunFactory& operator=(const RunFactory&); // stop default
            mutable boost::shared_ptr<fwlite::Run> run_;


            // ---------- member data --------------------------------
    };
}

#endif /*__CINT__ */

#endif
