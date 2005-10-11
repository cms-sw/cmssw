#ifndef Modules_EmptyESSource_h
#define Modules_EmptyESSource_h
// -*- C++ -*-
//
// Package:     Modules
// Class  :     EmptyESSource
// 
/**\class EmptyESSource EmptyESSource.h FWCore/Modules/src/EmptyESSource.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Jun 25 17:23:12 EDT 2005
// $Id: EmptyESSource.h,v 1.6 2005/10/03 23:23:09 chrjones Exp $
//

// system include files
#include <set>

// user include files
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

// forward declarations
namespace edm {
class ParameterSet;
   
class EmptyESSource : public  EventSetupRecordIntervalFinder
{

   public:
      EmptyESSource(const ParameterSet&);
      //virtual ~EmptyESSource();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
   void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                        const edm::IOVSyncValue& iTime, 
                        edm::ValidityInterval& oInterval);
      
   private:
      EmptyESSource(const EmptyESSource&); // stop default

      const EmptyESSource& operator=(const EmptyESSource&); // stop default
      
      // ---------- member data --------------------------------
      std::string recordName_;
      std::set <edm::IOVSyncValue> setOfIOV_;
      bool iovIsTime_;
};
}

#endif
