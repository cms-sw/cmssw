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
// $Id: CaloTPGTranscoderULUTs.cc,v 1.5 2009/10/27 12:08:49 kvtsang Exp $
//
//


// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

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
  bool read_Ascii_Compression;
  bool read_Ascii_RCT;
  std::vector<int> ietal;
  std::vector<int> ietah;
  std::vector<int> ZS;
  std::vector<int> LUTfactor;
  double nominal_gain;
  double RCTLSB;
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
   read_Ascii_Compression = false;
   read_Ascii_RCT = false;
   read_Ascii_Compression=iConfig.getParameter<bool>("read_Ascii_Compression_LUTs");
   read_Ascii_RCT=iConfig.getParameter<bool>("read_Ascii_RCT_LUTs");

   ietal = iConfig.getParameter< std::vector<int> >("ietaLowerBound");
   ietah = iConfig.getParameter< std::vector<int> >("ietaUpperBound");
   ZS = iConfig.getParameter< std::vector<int> >("ZS");
   LUTfactor = iConfig.getParameter< std::vector<int> >("LUTfactor");
   nominal_gain = iConfig.getParameter<double>("nominal_gain");
   RCTLSB = iConfig.getParameter<double>("RCTLSB");

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
   std::string file1="";
   std::string file2="";
   if (read_Ascii_RCT && read_Ascii_Compression) {
	 edm::LogInfo("Level1") << "Using " << hfilename1_.fullPath() << " & " << hfilename2_.fullPath()
			  << " for CaloTPGTranscoderULUTs HCAL initialization";
	 //std::auto_ptr<CaloTPGTranscoder> pTCoder(new CaloTPGTranscoderULUT(hfilename1_.fullPath(), hfilename2_.fullPath()));
	 //return pTCoder;
       file1 = hfilename1_.fullPath();
       file2 = hfilename2_.fullPath();
   } else if (read_Ascii_RCT && !read_Ascii_Compression) {
	 edm::LogInfo("Level1") << "Using analytical compression and " << hfilename2_.fullPath()
			  << " RCT decompression for CaloTPGTranscoderULUTs HCAL initialization";
	 //std::auto_ptr<CaloTPGTranscoder> pTCoder(new CaloTPGTranscoderULUT("", hfilename2_.fullPath()));
	 //return pTCoder;
       file2 = hfilename2_.fullPath();
   } else if (read_Ascii_Compression && !read_Ascii_RCT) {
	 edm::LogInfo("Level1") << "Using ASCII compression tables " << hfilename1_.fullPath()
			  << " and automatic RCT decompression for CaloTPGTranscoderULUTs HCAL initialization";
	 //std::auto_ptr<CaloTPGTranscoder> pTCoder(new CaloTPGTranscoderULUT(hfilename1_.fullPath(),""));
     //return pTCoder;
       file1 = hfilename1_.fullPath();
   } else {
	 edm::LogInfo("Level1") << "Using analytical compression and RCT decompression for CaloTPGTranscoderULUTs HCAL initialization";
	 //std::auto_ptr<CaloTPGTranscoder> pTCoder(new CaloTPGTranscoderULUT());
	 //return pTCoder;
   }
   //std::auto_ptr<CaloTPGTranscoder> pTCoder(new CaloTPGTranscoderULUT(ietal, ietah, ZS, LUTfactor, RCTLSB, nominal_gain, file1, file2));
   std::auto_ptr<CaloTPGTranscoder> pTCoder(new CaloTPGTranscoderULUT(file1, file2));
   return pTCoder;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(CaloTPGTranscoderULUTs);
