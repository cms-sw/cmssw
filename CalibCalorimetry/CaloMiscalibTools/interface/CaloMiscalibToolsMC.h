#ifndef _CALOMISCALIBTOOLSMC_H
#define _CALOMISCALIBTOOLSMC_H

// -*- C++ -*-
//
// Package:    CaloMiscalibToolsMC
// Class:      CaloMiscalibToolsMC
// 
/**\class CaloMiscalibToolsMC CaloMiscalibToolsMC.cc CalibCalorimetry/CaloMiscalibToolsMC/src/CaloMiscalibToolsMC.cc

 Description: Definition of CaloMiscalibToolsMC

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lorenzo AGOSTINO
//         Created:  Mon Jul 17 18:07:01 CEST 2006
// $Id: CaloMiscalibToolsMC.h,v 1.1 2009/04/08 22:29:38 fra Exp $
//
// Modified       : Luca Malgeri 
// Date:          : 11/09/2006 
// Reason         : split class definition (.h) from source code (.cc)
 

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/SourceFactory.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapEcal.h"

//
// class decleration
//

class CaloMiscalibToolsMC : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
   public:
      CaloMiscalibToolsMC(const edm::ParameterSet&);
      ~CaloMiscalibToolsMC();

      typedef const  EcalIntercalibConstantsMC * ReturnType;

      ReturnType produce(const EcalIntercalibConstantsMCRcd&);
   private:
      // ----------member data ---------------------------
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
    
    CaloMiscalibMapEcal map_;
    std::string barrelfile_; 
    std::string endcapfile_; 
    std::string barrelfileinpath_; 
    std::string endcapfileinpath_; 

};

#endif
