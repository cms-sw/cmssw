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
// $Id: PrintEventSetupContent.cc,v 1.1 2007/11/01 23:24:39 chrjones Exp $
//
//


// system include files
#include <memory>
#include <map>
// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Modules/src/EventSetupRecordDataGetter.h"

#include "FWCore/Framework/interface/EventSetup.h"//add by Yao
#include "FWCore/MessageLogger/interface/MessageLogger.h"
 
//
// class decleration
//


class PrintEventSetupContent : public edm::EDAnalyzer {
   public:
      explicit PrintEventSetupContent(const edm::ParameterSet&);
      ~PrintEventSetupContent();
     


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
       
      // ----------member data ---------------------------
std::map<edm::eventsetup::EventSetupRecordKey, unsigned long long > cacheIdentifiers_;
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
PrintEventSetupContent::PrintEventSetupContent(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is neededEventSetupRecordDataGetter::EventSetupRecordDataGetter(const edm::ParameterSet& iConfig):
  //  getter = new edm::EventSetupRecordDataGetter::EventSetupRecordDataGetter(iConfig);
}


PrintEventSetupContent::~PrintEventSetupContent()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
PrintEventSetupContent::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{ 
  typedef std::vector<edm::eventsetup::EventSetupRecordKey> Records;
  typedef std::vector<edm::eventsetup::DataKey> Data;
  
  Records records;
  Data data;
  iSetup.fillAvailableRecordKeys(records);
  int iflag=0;
  
  
  for(Records::iterator itrecords = records.begin(), itrecordsend = records.end();
     itrecords != itrecordsend; ++itrecords ) {
    
    const edm::eventsetup::EventSetupRecord* rec = iSetup.find(*itrecords);
    
    
    
    if( 0 != rec && cacheIdentifiers_[*itrecords] != rec->cacheIdentifier() ) {
      iflag++;
	if(iflag==1)
	  edm::LogSystem("ESContent")<<"\n"<<"Changed Record"<<"\n  "<<"<datatype>"<<" "<<"'label'"; 
      cacheIdentifiers_[*itrecords] = rec->cacheIdentifier();
      edm::LogAbsolute("ESContent")<<itrecords->name()<<std::endl;
     
      rec->fillRegisteredDataKeys(data);
      for(Data::iterator itdata = data.begin(), itdataend = data.end(); itdata != itdataend; ++itdata){
	edm::LogAbsolute("ESContent")<<"  "<<itdata->type().name()<<" '"<<itdata->name().value()<<"'"<<std::endl;
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
//}


// ------------ method called once each job just before starting event loop  ------------
void 
PrintEventSetupContent::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PrintEventSetupContent::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(PrintEventSetupContent);
