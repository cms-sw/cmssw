// -*- C++ -*-
//
// Package:    CaloMiscalibToolsMC
// Class:      CaloMiscalibToolsMC
// 
/**\class CaloMiscalibToolsMC CaloMiscalibToolsMC.h CalibCalorimetry/CaloMiscalibToolsMC/interface/CaloMiscalibToolsMC.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lorenzo AGOSTINO
//         Created:  Wed May 31 10:37:45 CEST 2006
// $Id: CaloMiscalibToolsMC.cc,v 1.1 2009/04/08 22:29:39 fra Exp $
//
// Modified       : Luca Malgeri 
// Date:          : 11/09/2006 
// Reason         : split class definition (.h) from source code (.cc)
//
//


// system include files

// user include files
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibToolsMC.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalBarrel.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalEndcap.h"

//
// constructors and destructor
//
CaloMiscalibToolsMC::CaloMiscalibToolsMC(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
  map_.prefillMap();

  barrelfileinpath_=iConfig.getUntrackedParameter<std::string> ("fileNameBarrel","");
  endcapfileinpath_=iConfig.getUntrackedParameter<std::string> ("fileNameEndcap","");

  edm::FileInPath barrelfiletmp("CalibCalorimetry/CaloMiscalibTools/data/"+barrelfileinpath_);
  edm::FileInPath endcapfiletmp("CalibCalorimetry/CaloMiscalibTools/data/"+endcapfileinpath_);
  
  
  barrelfile_=barrelfiletmp.fullPath();
  endcapfile_=endcapfiletmp.fullPath();

  std::cout <<"Barrel file is:"<< barrelfile_<<std::endl;
  std::cout <<"endcap file is:"<< endcapfile_<<std::endl;


   // added by Zhen (changed since 1_2_0)
   setWhatProduced(this,&CaloMiscalibToolsMC::produce);
   findingRecord<EcalIntercalibConstantsMCRcd>();
   //now do what ever other initialization is needed
}


CaloMiscalibToolsMC::~CaloMiscalibToolsMC()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
CaloMiscalibToolsMC::ReturnType
CaloMiscalibToolsMC::produce(const EcalIntercalibConstantsMCRcd& iRecord)
{
    map_.prefillMap();
    MiscalibReaderFromXMLEcalBarrel barrelreader_(map_);
    MiscalibReaderFromXMLEcalEndcap endcapreader_(map_);
    if(!barrelfile_.empty()) barrelreader_.parseXMLMiscalibFile(barrelfile_);
    if(!endcapfile_.empty())endcapreader_.parseXMLMiscalibFile(endcapfile_);
    map_.print();
    // Added by Zhen, need a new object so to not be deleted at exit
    //    std::cout<<"about to copy"<<std::endl;
    EcalIntercalibConstantsMC* mydata=new EcalIntercalibConstantsMC(map_.get());
    //    std::cout<<"mydata "<<mydata<<std::endl;
    return mydata;
}

 void CaloMiscalibToolsMC::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }

