
// -*- C++ -*-
//
// Package:    PrintEventSetupContent
// Class:      PrintEventSetupContent
// 
/**\class PrintEventSetupContent PrintEventSetupContent.cc GetRecordName/PrintEventSetupContent/src/PrintEventSetupContent.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Weng Yao
//         Created:  Tue Oct  2 13:49:56 EDT 2007
//
//


// system include files
#include <memory>
#include <map>
#include <iostream>
// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "DataFormats/Provenance/interface/EventID.h" 
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
//
// class decleration
//


namespace edm {
  class PrintEventSetupContent : public EDAnalyzer {
    public:
      explicit PrintEventSetupContent(ParameterSet const&);
      ~PrintEventSetupContent();
     
      static void fillDescriptions(ConfigurationDescriptions& descriptions);


    private:
      virtual void beginJob();

      virtual void analyze(Event const&, EventSetup const&);
      virtual void endJob() ;
      virtual void beginRun(Run const&, EventSetup const&);
      virtual void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&);

      void print(EventSetup const&);
       
      // ----------member data ---------------------------
  std::map<eventsetup::EventSetupRecordKey, unsigned long long > cacheIdentifiers_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
  PrintEventSetupContent::PrintEventSetupContent(ParameterSet const& iConfig) {
  //now do what ever initialization is neededEventSetupRecordDataGetter::EventSetupRecordDataGetter(ParameterSet const& iConfig):
  //  getter = new EventSetupRecordDataGetter::EventSetupRecordDataGetter(iConfig);
  }


  PrintEventSetupContent::~PrintEventSetupContent() {
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

  }


  //
  // member functions
  //
  
  // ------------ method called to for each event  ------------
  void
  PrintEventSetupContent::analyze(Event const& iEvent, EventSetup const& iSetup) { 
    print(iSetup);
  }

  void 
  PrintEventSetupContent::beginRun(Run const&, EventSetup const& iSetup){
    print(iSetup);
  }

  void 
  PrintEventSetupContent::beginLuminosityBlock(LuminosityBlock const&, EventSetup const& iSetup){
    print(iSetup);
  }

  void
  PrintEventSetupContent::print (EventSetup const& iSetup) { 
    typedef std::vector<eventsetup::EventSetupRecordKey> Records;
    typedef std::vector<eventsetup::DataKey> Data;
    
    Records records;
    Data data;
    iSetup.fillAvailableRecordKeys(records);
    int iflag=0;
    
    
    for(Records::iterator itrecords = records.begin(), itrecordsend = records.end();
       itrecords != itrecordsend; ++itrecords ) {
      
      eventsetup::EventSetupRecord const* rec = iSetup.find(*itrecords);
      
      
      
      if(0 != rec && cacheIdentifiers_[*itrecords] != rec->cacheIdentifier() ) {
        ++iflag;
  	if(iflag==1)
  	  LogSystem("ESContent")<<"\n"<<"Changed Record"<<"\n  "<<"<datatype>"<<" "<<"'label'"; 
        cacheIdentifiers_[*itrecords] = rec->cacheIdentifier();
        LogAbsolute("ESContent")<<itrecords->name()<<std::endl;
  
        LogAbsolute("ESContent")<<" start: "<<rec->validityInterval().first().eventID()<<" time: "<<rec->validityInterval().first().time().value()<<std::endl;
        LogAbsolute("ESContent")<<" end:   "<<rec->validityInterval().last().eventID()<<" time: "<<rec->validityInterval().last().time().value()<<std::endl;
        rec->fillRegisteredDataKeys(data);
        for(Data::iterator itdata = data.begin(), itdataend = data.end(); itdata != itdataend; ++itdata){
  	LogAbsolute("ESContent")<<"  "<<itdata->type().name()<<" '"<<itdata->name().value()<<"'"<<std::endl;
        }         
      }   
    }
  }
  
   //#ifdef THIS_IS_AN_EVENT_EXAMPLE
   //   Handle<ExampleData> pIn;
   //   iEvent.getByLabel("example",pIn);
   //#endif

   //#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   //   ESHandle<SetupData> pSetup;
   //   iSetup.get<SetupRecord>().get(pSetup);
   //#endif
  
  
  // ------------ method called once each job just before starting event loop  ------------
  void 
  PrintEventSetupContent::beginJob() {
  }
  
  // ------------ method called once each job just after ending the event loop  ------------
  void 
  PrintEventSetupContent::endJob() {
  }

  // ------------ method called once each job for validation  ------------
  void
  PrintEventSetupContent::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    descriptions.add("PrintEventSetupContent", desc);
  }
}
  
//define this as a plug-in
using edm::PrintEventSetupContent;
DEFINE_FWK_MODULE(PrintEventSetupContent);
