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
// $Id: CaloMiscalibTools.cc,v 1.4 2008/03/26 14:07:42 fra Exp $
//
// Modified       : Luca Malgeri 
// Date:          : 11/09/2006 
// Reason         : split class definition (.h) from source code (.cc)
//
//


// system include files

// user include files
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibTools.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondTools/Ecal/interface/EcalIntercalibConstantsXMLTranslator.h"
//
// constructors and destructor
//
CaloMiscalibTools::CaloMiscalibTools(const edm::ParameterSet& iConfig)
{

  constantsfile_=iConfig.getUntrackedParameter<std::string> ("fileName","");

  edm::FileInPath filetmp("CalibCalorimetry/CaloMiscalibTools/data/"+constantsfile_);
  
  
  constantsfile_=filetmp.fullPath();


  edm::LogInfo("CaloMiscalibTools: using")<<constantsfile_;

   // added by Zhen (changed since 1_2_0)
   setWhatProduced(this,&CaloMiscalibTools::produce);
   findingRecord<EcalIntercalibConstantsRcd>();
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
   
  
  EcalCondHeader h;
  EcalIntercalibConstants rcd ;
  int ret=EcalIntercalibConstantsXMLTranslator::readXML(constantsfile_,h,rcd);

  if (ret)   edm::LogError("CaloMiscalibReader: cannot parse xml file");

  EcalIntercalibConstants* mydata=new EcalIntercalibConstants(rcd);
    //    std::cout<<"mydata "<<mydata<<std::endl;
  return mydata;
}

 void CaloMiscalibTools::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }

