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
#include "CondFormats/CSCObjects/interface/CSCPedestals.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakePedestalsConditions.h"
#include "CondFormats/DataRecord/interface/CSCPedestalsRcd.h"

CSCFakePedestalsConditions::CSCFakePedestalsConditions(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  pedestals.prefillPedestalsMap();
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this,&CSCFakePedestalsConditions::producePedestals);
  findingRecord<CSCPedestalsRcd>();
  //now do what ever other initialization is needed
}


CSCFakePedestalsConditions::~CSCFakePedestalsConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakePedestalsConditions::ReturnType
CSCFakePedestalsConditions::producePedestals(const CSCPedestalsRcd& iRecord)
{
    pedestals.prefillPedestalsMap();
    //    pedestals.print();
    // Added by Zhen, need a new object so to not be deleted at exit
    //    std::cout<<"about to copy"<<std::endl;
    CSCPedestals* mydata=new CSCPedestals(pedestals.get());
    
    /*
    std::cout <<"=========================DUMP from produce=====================" << std::endl;
    std::map<int,std::vector<CSCPedestals::Item> >::const_iterator it;
    for( it=mydata->pedestals.begin();it!=mydata->pedestals.end(); ++it ){
       std::cout<<"layer id found "<<it->first<<std::endl;
       std::vector<CSCPedestals::Item>::const_iterator pedestalsit;
       for( pedestalsit=it->second.begin(); pedestalsit!=it->second.end(); ++pedestalsit ){
         std::cout << "  pedestals:  " <<pedestalsit->ped << " intercept: " << pedestalsit->rms
                   << std::endl;
       }
    }
    std::cout <<"=========================END DUMP from produce=====================" << std::endl;
    */
       return mydata;

}

 void CSCFakePedestalsConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
