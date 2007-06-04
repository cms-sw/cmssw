#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

//FW include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

//CSCObjects
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeGainsConditions.h"
#include "CondFormats/DataRecord/interface/CSCGainsRcd.h"

CSCFakeGainsConditions::CSCFakeGainsConditions(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  gains.prefillGainsMap();
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this,&CSCFakeGainsConditions::produceGains);
  findingRecord<CSCGainsRcd>();
  //now do what ever other initialization is needed
}


CSCFakeGainsConditions::~CSCFakeGainsConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakeGainsConditions::ReturnType
CSCFakeGainsConditions::produceGains(const CSCGainsRcd& iRecord)
{
    gains.prefillGainsMap();
    //    gains.print();
    // Added by Zhen, need a new object so to not be deleted at exit
    //    std::cout<<"about to copy"<<std::endl;
    CSCGains* mydata=new CSCGains(gains.get());
    
    /*
    std::cout <<"=========================DUMP from produce=====================" << std::endl;
    std::map<int,std::vector<CSCGains::Item> >::const_iterator it;
    for( it=mydata->gains.begin();it!=mydata->gains.end(); ++it ){
       std::cout<<"layer id found "<<it->first<<std::endl;
       std::vector<CSCGains::Item>::const_iterator gainsit;
       for( gainsit=it->second.begin(); gainsit!=it->second.end(); ++gainsit ){
         std::cout << "  gains:  " <<gainsit->gain_slope << " intercept: " << gainsit->gain_intercept
                   << std::endl;
       }
    }
    std::cout <<"=========================END DUMP from produce=====================" << std::endl;
    */
       return mydata;

}

 void CSCFakeGainsConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
