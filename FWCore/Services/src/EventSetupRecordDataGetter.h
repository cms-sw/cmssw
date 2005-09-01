#ifndef Services_EventSetupRecordDataGetter_h
#define Services_EventSetupRecordDataGetter_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     EventSetupRecordDataGetter
// 
/**\class EventSetupRecordDataGetter EventSetupRecordDataGetter.h FWCore/Services/interface/EventSetupRecordDataGetter.h

 Description: Can be configured to 'get' any Data in any EventSetup Record.  Primarily used for testing.

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jun 28 13:45:06 EDT 2005
// $Id: EventSetupRecordDataGetter.h,v 1.3 2005/08/04 15:04:11 chrjones Exp $
//

// system include files
#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"

// forward declarations
namespace edm {
   class EventSetupRecordDataGetter : public edm::EDAnalyzer {
public:
      explicit EventSetupRecordDataGetter(const edm::ParameterSet&);
      ~EventSetupRecordDataGetter();
      
      
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
private:
         // ----------member data ---------------------------
      ParameterSet pSet_;
      
      typedef std::map<edm::eventsetup::EventSetupRecordKey, std::vector<edm::eventsetup::DataKey> > RecordToDataKeys;
      RecordToDataKeys recordToDataKeys_;
      //NOTE: forced to use shared_ptr<IOVSyncValue> instead of just IOVSyncValue since insertion into the map
      //  requires the presence of a default constructor which IOVSyncValue does not have
      std::map<eventsetup::EventSetupRecordKey, boost::shared_ptr<edm::IOVSyncValue> > recordToIOVSyncValue_;
      bool verbose_;
   };
}


#endif
