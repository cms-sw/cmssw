// -*- C++ -*-
//
// Package:    CaloTPGTranscoderULUTs
// Class:      CaloTPGTranscoderULUTs
// 
/**\class CaloTPGTranscoderULUTs CaloTPGTranscoderULUTs.h src/CaloTPGTranscoderULUTs/interface/CaloTPGTranscoderULUTs.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Fri Sep 15 11:49:44 CDT 2006
// $Id: CaloTPGTranscoderULUTs.cc,v 1.3 2006/12/21 03:18:43 dasu Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


//
// class decleration
//

class CaloTPGTranscoderULUTs : public edm::ESProducer,
			 public edm::EventSetupRecordIntervalFinder {
public:
  CaloTPGTranscoderULUTs(const edm::ParameterSet&);
  ~CaloTPGTranscoderULUTs();
  
  typedef std::auto_ptr<CaloTPGTranscoder> ReturnType;
  
  ReturnType produce(const CaloTPGRecord&);

  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey, const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval ) {
    oInterval = edm::ValidityInterval (edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime()); //infinite
  }
private:
  // ----------member data ---------------------------
  edm::FileInPath hfilename1_;
  edm::FileInPath hfilename2_;
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
CaloTPGTranscoderULUTs::CaloTPGTranscoderULUTs(const edm::ParameterSet& iConfig) :
  hfilename1_(iConfig.getParameter<edm::FileInPath>("hcalLUT1")),
  hfilename2_(iConfig.getParameter<edm::FileInPath>("hcalLUT2"))
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);
   findingRecord<CaloTPGRecord>();

   //now do what ever other initialization is needed
}


CaloTPGTranscoderULUTs::~CaloTPGTranscoderULUTs()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
CaloTPGTranscoderULUTs::ReturnType
CaloTPGTranscoderULUTs::produce(const CaloTPGRecord& iRecord)
{
   using namespace edm::es;
   edm::LogInfo("Level1") << "Using " << hfilename1_.fullPath() << " & " << hfilename2_.fullPath()
			  << " for CaloTPGTranscoderULUTs HCAL initialization";
   std::auto_ptr<CaloTPGTranscoder> pTCoder(new CaloTPGTranscoderULUT(hfilename1_.fullPath(), hfilename2_.fullPath()));

   return pTCoder ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(CaloTPGTranscoderULUTs);
