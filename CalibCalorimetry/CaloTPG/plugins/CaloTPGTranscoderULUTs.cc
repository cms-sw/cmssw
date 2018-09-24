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
//
//


// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/HcalObjects/interface/HcalLutMetadata.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

//
// class decleration
//

class CaloTPGTranscoderULUTs : public edm::ESProducer {
   public:
      CaloTPGTranscoderULUTs(const edm::ParameterSet&);
      ~CaloTPGTranscoderULUTs() override;

      typedef std::unique_ptr<CaloTPGTranscoder> ReturnType;

      ReturnType produce(const CaloTPGRecord&);

   private:
      // ----------member data ---------------------------
      const edm::FileInPath hfilename1_;
      const edm::FileInPath hfilename2_;
      const bool read_Ascii_Compression;
      const bool read_Ascii_RCT;
      const std::vector<int> ietal;
      const std::vector<int> ietah;
      const std::vector<int> ZS;
      const std::vector<int> LUTfactor;
      const bool linearLUTs_;
      const double nominal_gain;
      const double RCTLSB;
      const int NCTScaleShift;
      const int RCTScaleShift;
      const double lsbQIE8;
      const double lsbQIE11;
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
   hfilename2_(iConfig.getParameter<edm::FileInPath>("hcalLUT2")),
   read_Ascii_Compression(iConfig.getParameter<bool>("read_Ascii_Compression_LUTs")),
   read_Ascii_RCT(iConfig.getParameter<bool>("read_Ascii_RCT_LUTs")),
   ietal(iConfig.getParameter<std::vector<int>>("ietaLowerBound")),
   ietah(iConfig.getParameter<std::vector<int>>("ietaUpperBound")),
   ZS(iConfig.getParameter<std::vector<int>>("ZS")),
   LUTfactor(iConfig.getParameter<std::vector<int>>("LUTfactor")),
   linearLUTs_(iConfig.getParameter<bool>("linearLUTs")),
   nominal_gain(iConfig.getParameter<double>("nominal_gain")),
   RCTLSB(iConfig.getParameter<double>("RCTLSB")),
   NCTScaleShift(iConfig.getParameter<edm::ParameterSet>("tpScales").getParameter<edm::ParameterSet>("HF").getParameter<int>("NCTShift")),
   RCTScaleShift(iConfig.getParameter<edm::ParameterSet>("tpScales").getParameter<edm::ParameterSet>("HF").getParameter<int>("RCTShift")),
   lsbQIE8(iConfig.getParameter<edm::ParameterSet>("tpScales").getParameter<edm::ParameterSet>("HBHE").getParameter<double>("LSBQIE8")),
   lsbQIE11(iConfig.getParameter<edm::ParameterSet>("tpScales").getParameter<edm::ParameterSet>("HBHE").getParameter<double>("LSBQIE11"))
{
   setWhatProduced(this);
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
	 //std::unique_ptr<CaloTPGTranscoder> pTCoder(new CaloTPGTranscoderULUT(hfilename1_.fullPath(), hfilename2_.fullPath()));
	 //return pTCoder;
       file1 = hfilename1_.fullPath();
       file2 = hfilename2_.fullPath();
   } else if (read_Ascii_RCT && !read_Ascii_Compression) {
	 edm::LogInfo("Level1") << "Using analytical compression and " << hfilename2_.fullPath()
			  << " RCT decompression for CaloTPGTranscoderULUTs HCAL initialization";
	 //std::unique_ptr<CaloTPGTranscoder> pTCoder(new CaloTPGTranscoderULUT("", hfilename2_.fullPath()));
	 //return pTCoder;
       file2 = hfilename2_.fullPath();
   } else if (read_Ascii_Compression && !read_Ascii_RCT) {
	 edm::LogInfo("Level1") << "Using ASCII compression tables " << hfilename1_.fullPath()
			  << " and automatic RCT decompression for CaloTPGTranscoderULUTs HCAL initialization";
	 //std::unique_ptr<CaloTPGTranscoder> pTCoder(new CaloTPGTranscoderULUT(hfilename1_.fullPath(),""));
     //return pTCoder;
       file1 = hfilename1_.fullPath();
   } else {
	 edm::LogInfo("Level1") << "Using analytical compression and RCT decompression for CaloTPGTranscoderULUTs HCAL initialization";
	 //std::unique_ptr<CaloTPGTranscoder> pTCoder(new CaloTPGTranscoderULUT());
	 //return pTCoder;
   }
   //std::unique_ptr<CaloTPGTranscoder> pTCoder(new CaloTPGTranscoderULUT(ietal, ietah, ZS, LUTfactor, RCTLSB, nominal_gain, file1, file2));
   
   edm::ESHandle<HcalLutMetadata> lutMetadata;
   iRecord.getRecord<HcalLutMetadataRcd>().get(lutMetadata);
   edm::ESHandle<HcalTrigTowerGeometry> theTrigTowerGeometry;
   iRecord.getRecord<CaloGeometryRecord>().get(theTrigTowerGeometry);

   edm::ESHandle<HcalTopology> htopo;
   iRecord.getRecord<HcalLutMetadataRcd>().getRecord<HcalRecNumberingRecord>().get(htopo);

   HcalLutMetadata fullLut{ *lutMetadata };
   fullLut.setTopo(htopo.product());

   std::unique_ptr<CaloTPGTranscoderULUT> pTCoder(new CaloTPGTranscoderULUT(file1, file2));
   pTCoder->setup(fullLut, *theTrigTowerGeometry, NCTScaleShift, RCTScaleShift, lsbQIE8, lsbQIE11, linearLUTs_);
   return std::unique_ptr<CaloTPGTranscoder>( std::move(pTCoder) );
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(CaloTPGTranscoderULUTs);
