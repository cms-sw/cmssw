// -*- C++ -*-
//
// Package:    EventSetupRecordDataGetter
// Class:      EventSetupRecordDataGetter
// 
/**\class EventSetupRecordDataGetter EventSetupRecordDataGetter.cc src/EventSetupRecordDataGetter/src/EventSetupRecordDataGetter.cc

 Description: Can be configured to 'get' any Data in any EventSetup Record.  Primarily used for testing.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jun 28 11:10:24 EDT 2005
//
//


// system include files
#include <map>
#include <vector>
#include <memory>
#include <set>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//
// class decleration
//
namespace edm {
   class EventSetupRecordDataGetter : public EDAnalyzer {
public:
     explicit EventSetupRecordDataGetter(ParameterSet const&);
     ~EventSetupRecordDataGetter();
      
      
     virtual void analyze(Event const&, EventSetup const&);
     virtual void beginRun(Run const&, EventSetup const&);
     virtual void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&);

     static void fillDescriptions(ConfigurationDescriptions& descriptions);

private:
     void doGet(EventSetup const&);
        // ----------member data ---------------------------
     ParameterSet pSet_;
      
     typedef std::map<eventsetup::EventSetupRecordKey, std::vector<eventsetup::DataKey> > RecordToDataKeys;
     RecordToDataKeys recordToDataKeys_;
     std::map<eventsetup::EventSetupRecordKey, unsigned long long> recordToCacheIdentifier_;
     bool verbose_;

  };

//
// constructors and destructor
//
   EventSetupRecordDataGetter::EventSetupRecordDataGetter(ParameterSet const& iConfig) :
    pSet_(iConfig),
    recordToDataKeys_(),
    recordToCacheIdentifier_(),
    verbose_(iConfig.getUntrackedParameter<bool>("verbose")) {}

   EventSetupRecordDataGetter::~EventSetupRecordDataGetter() {

     // do anything here that needs to be done at desctruction time
     // (e.g. close files, deallocate resources etc.)

   }


//
// member functions
//

// ------------ method called to produce the data  ------------
   void EventSetupRecordDataGetter::fillDescriptions(ConfigurationDescriptions& descriptions) {
      descriptions.setComment("Retrieves specified data from the EventSetup sytem whenever that data changes.");
      
      ParameterSetDescription desc;
      desc.addOptionalUntracked<bool>("verbose", false)->setComment("Print a message to the logger each time a data item is gotten.");

      ParameterSetDescription toGet;
      toGet.add<std::string>("record")->setComment("The name of an EventSetup record holding the data you want obtained.");
      toGet.add<std::vector<std::string> >("data")->setComment("The identifier for the data you wish to retrieve. " 
                                                               "The identifier is in two parts separated by a backslash '/'. "
                                                               "The first part is the C++ class name of the data and the "
                                                               "second part is the label used when getting the data (blank is acceptable). "
                                                               "If there is no label, the backslash may be omitted."
      );
      
      desc.addVPSet("toGet", toGet)->setComment("The contained PSets must have the following structure.\n"
                                                "A 'string' named 'record' that holds the name of an EventSetup record holding the data you want to obtain.\n"
                                                "a 'vstring' named 'data' that holds identifiers for the data you wish to retrieve. "
                                                "The identifier is in two parts separated by a backslash '/'. "
                                                "The first part is the C++ class name of the data and the "
                                                "second part is the label used when getting the data (blank is acceptable). "
                                                "If there is no label, the backslash may be omitted."
      );
      descriptions.add("getEventSetupData", desc);
   }

   void 
   EventSetupRecordDataGetter::beginRun(Run const&, EventSetup const& iSetup) {
      doGet(iSetup);
   }

   void 
   EventSetupRecordDataGetter::beginLuminosityBlock(LuminosityBlock const&, EventSetup const& iSetup) {
      doGet(iSetup);
   }
   
   void
   EventSetupRecordDataGetter::analyze(edm::Event const& /*iEvent*/, edm::EventSetup const& iSetup) {
      doGet(iSetup);
   }
   
   void
   EventSetupRecordDataGetter::doGet(EventSetup const& iSetup) {  
      if(0 == recordToDataKeys_.size()) {
         typedef std::vector<ParameterSet> Parameters;
         Parameters const& toGet = pSet_.getParameterSetVector("toGet");
         
         for(Parameters::const_iterator itToGet = toGet.begin(), itToGetEnd = toGet.end(); itToGet != itToGetEnd; ++itToGet) {
            std::string recordName = itToGet->getParameter<std::string>("record");
            
            eventsetup::EventSetupRecordKey recordKey(eventsetup::EventSetupRecordKey::TypeTag::findType(recordName));
            if(recordKey.type() == eventsetup::EventSetupRecordKey::TypeTag()) {
              //record not found
              edm::LogWarning("DataGetter") <<"Record \""<< recordName <<"\" does not exist "<<std::endl;
               
               continue;
            }
            typedef std::vector<std::string> Strings;
            Strings dataNames = itToGet->getParameter<Strings>("data");
            std::vector<eventsetup::DataKey> dataKeys;
            for(Strings::iterator itDatum = dataNames.begin(), itDatumEnd = dataNames.end();
                  itDatum != itDatumEnd; ++itDatum) {
               std::string datumName(*itDatum, 0, itDatum->find_first_of("/"));
               std::string labelName;
               if(itDatum->size() != datumName.size()) {
                  labelName = std::string(*itDatum, datumName.size()+1);
               }
               eventsetup::TypeTag datumType = eventsetup::TypeTag::findType(datumName);
               if(datumType == eventsetup::TypeTag()) {
                 //not found
                 edm::LogWarning("DataGetter") <<"data item of type \""<< datumName <<"\" does not exist"<<std::endl;

                
                  continue;
               }
               eventsetup::DataKey datumKey(datumType, labelName.c_str());
               dataKeys.push_back(datumKey); 
            }
            recordToDataKeys_.insert(std::make_pair(recordKey, dataKeys));
            recordToCacheIdentifier_.insert(std::make_pair(recordKey, 0));
         }
         if(toGet.empty()) {
            //This means we should get everything in the EventSetup
            std::vector<eventsetup::EventSetupRecordKey> recordKeys;
            iSetup.fillAvailableRecordKeys(recordKeys);
            std::vector<eventsetup::DataKey> dataKeys;

            for(std::vector<eventsetup::EventSetupRecordKey>::iterator itRKey = recordKeys.begin(), itRKeyEnd = recordKeys.end();
                itRKey != itRKeyEnd;
                ++itRKey) {               
               eventsetup::EventSetupRecord const* record = iSetup.find(*itRKey);
               dataKeys.clear();
               record->fillRegisteredDataKeys(dataKeys);
               recordToDataKeys_.insert(std::make_pair(*itRKey, dataKeys));
               recordToCacheIdentifier_.insert(std::make_pair(*itRKey, 0));               
            }
         }
      }
      
      using namespace edm::eventsetup;

      //For each requested Record get the requested data only if the Record is in a new IOV
      
      for(RecordToDataKeys::iterator itRecord = recordToDataKeys_.begin(), itRecordEnd = recordToDataKeys_.end();
           itRecord != itRecordEnd;
           ++itRecord) {
         EventSetupRecord const* pRecord = iSetup.find(itRecord->first);
         if(0 == pRecord) {
           edm::LogWarning("RecordNotInIOV") <<"The EventSetup Record '"<<itRecord->first.name()<<"' is not available for this IOV.";
         }
         if(0 != pRecord && pRecord->cacheIdentifier() != recordToCacheIdentifier_[itRecord->first]) {
            recordToCacheIdentifier_[itRecord->first] = pRecord->cacheIdentifier();
            typedef std::vector<DataKey> Keys;
            Keys const& keys = itRecord->second;
            for(Keys::const_iterator itKey = keys.begin(), itKeyEnd = keys.end();
                 itKey != itKeyEnd;
                 ++itKey) {
               if(! pRecord->doGet(*itKey)) {
                 edm::LogWarning("DataGetter") << "No data of type \""<<itKey->type().name() <<"\" with name \""<< itKey->name().value()<<"\" in record "<<itRecord->first.type().name() <<" found "<< std::endl;
               } else {
                  if(verbose_) {
                    edm::LogSystem("DataGetter") << "got data of type \""<<itKey->type().name() <<"\" with name \""<< itKey->name().value()<<"\" in record "<<itRecord->first.type().name() << std::endl;
                  }
               }
            }
         }
      }
   }
}
using edm::EventSetupRecordDataGetter;
DEFINE_FWK_MODULE(EventSetupRecordDataGetter);
