// -*- C++ -*-
//
// Package:    CaloMiscalibTools
// Class:      CaloMiscalibTools
// 
/**\class CaloMiscalibTools CaloMiscalibTools.h CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibTools.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lorenzo AGOSTINO
//         Created:  Wed May 31 10:37:45 CEST 2006
// $Id$
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalBarrel.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalEndcap.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapEcal.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"




//
// class decleration
//

class CaloMiscalibTools : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
   public:
      CaloMiscalibTools(const edm::ParameterSet&);
      ~CaloMiscalibTools();

      typedef const  EcalIntercalibConstants * ReturnType;

      ReturnType produce(const EcalIntercalibConstantsRcd&);
   private:
      // ----------member data ---------------------------
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
    
    CaloMiscalibMapEcal map_;
    std::string barrelfile_; 
    std::string endcapfile_; 
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
CaloMiscalibTools::CaloMiscalibTools(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   map_.prefillMap();
   barrelfile_=iConfig.getUntrackedParameter<std::string> ("fileNameBarrel","");
   endcapfile_=iConfig.getUntrackedParameter<std::string> ("fileNameEndcap","");
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


CaloMiscalibTools::~CaloMiscalibTools()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
CaloMiscalibTools::ReturnType
CaloMiscalibTools::produce(const EcalIntercalibConstantsRcd& iRecord)
{
    map_.prefillMap();
    MiscalibReaderFromXMLEcalBarrel barrelreader_(map_);
    MiscalibReaderFromXMLEcalEndcap endcapreader_(map_);
    if(!barrelfile_.empty()) barrelreader_.parseXMLMiscalibFile(barrelfile_);
    if(!endcapfile_.empty())endcapreader_.parseXMLMiscalibFile(endcapfile_);
    map_.print();
    return & (map_.get());
}

 void CaloMiscalibTools::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(CaloMiscalibTools)
