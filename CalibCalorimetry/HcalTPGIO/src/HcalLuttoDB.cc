// -*- C++ -*-
//
// Package:    HcalLuttoDB
// Class:      HcalLuttoDB
// 
/**\class HcalLuttoDB HcalLuttoDB.cc CalibCalorimetry/HcalLuttoDB/src/HcalLuttoDB.cc

 Description: <one line class summary>
 R
 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Michael Weinberger
//         Created:  Mon Mar 19 11:53:56 CDT 2007
// $Id: HcalLuttoDB.cc,v 1.6 2009/12/17 21:20:49 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"

using namespace edm;
using namespace std;
#include <iostream>
#include <fstream>
#include "FWCore/Utilities/interface/md5.h"

//
// class decleration
//

class HcalLuttoDB : public edm::EDAnalyzer {
public:
  explicit HcalLuttoDB(const edm::ParameterSet&);
  ~HcalLuttoDB();
  
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

private:
  void writeoutlut1(HcalDetId id, HcalElectronicsId eid, const std::vector<unsigned short>& lut, std::ostream& os);
  std::vector<unsigned char> extractOutputLut(const CaloTPGTranscoder& coder, HcalTrigTowerDetId chan);
  void writeoutlut2(HcalTrigTowerDetId id, HcalElectronicsId eid, const std::vector<unsigned char>& lut, std::ostream& os);
  bool filePerCrate_;
  std::string creationstamp_;
  std::string fileformat_;
  std::ostream* openPerCrate(int crate);
  std::ostream* openPerLut1(HcalElectronicsId eid);
  std::ostream* openPerLut2(HcalElectronicsId eid);
  std::ostream* openChecksums();
  std::ostream* oc_;
};      
      // ----------member data ---------------------------
 
std::string creationtag_;
std::string targetfirmware_;
static const int formatRevision_=1;

// constructors and destructor
//
HcalLuttoDB::HcalLuttoDB(const edm::ParameterSet& iConfig)
{
  
  creationtag_       = iConfig.getParameter<std::string>("creationtag");
  targetfirmware_    = iConfig.getParameter<std::string>("targetfirmware");
  filePerCrate_ = iConfig.getUntrackedParameter<bool>("filePerCrate",true);
  fileformat_=iConfig.getParameter<std::string>("filePrefix");
}

HcalLuttoDB::~HcalLuttoDB()
{
}


//
// member functions
//

std::ostream* HcalLuttoDB::openChecksums() {
  char fname[1024];
  snprintf(fname,1024,"%s_checksums.xml",fileformat_.c_str());
  std::ostream* os=new std::ofstream(fname);
  (*os) << "<?xml version=\"1.0\"?>\n<CFGBrick>\n";
  return os;
}

std::ostream* HcalLuttoDB::openPerCrate(int crate) {
  char fname[1024];
  snprintf(fname,1024,"%s_%d.xml",fileformat_.c_str(),crate);
  std::ostream* os=new std::ofstream(fname);
  (*os) << "<?xml version=\"1.0\"?>\n<CFGBrickSet>\n";
  return os;
}

std::ostream* HcalLuttoDB::openPerLut1(HcalElectronicsId eid) {
  char fname[1024];
  snprintf(fname,1024,"%s_%d_%d%c_%d_%d_1.xml",fileformat_.c_str(),eid.readoutVMECrateId(),
	   eid.htrSlot(),((eid.htrTopBottom())?('t'):('b')),eid.fiberIndex(),eid.fiberChanId());
  std::ostream* os=new std::ofstream(fname);
  (*os) << "<?xml version=\"1.0\"?>\n";
  return os;
}

std::ostream* HcalLuttoDB::openPerLut2(HcalElectronicsId eid) {
  char fname[1024];
  snprintf(fname,1024,"%s_%d_%d%c_%d_%d_2.xml",fileformat_.c_str(),eid.readoutVMECrateId(),
	   eid.htrSlot(),((eid.htrTopBottom())?('t'):('b')),eid.slbSiteNumber(),eid.slbChannelIndex());
  std::ostream* os=new std::ofstream(fname);
  (*os) << "<?xml version=\"1.0\"?>\n";
  return os;
}


void
HcalLuttoDB::writeoutlut1(HcalDetId id, HcalElectronicsId eid, const std::vector<unsigned short>& lut, std::ostream& os) {

  os <<"<CFGBrick> "<<std::endl;
  os <<" <Parameter name='IETA' type='int'>"<<id.ieta()<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='IPHI' type='int'>"<<id.iphi()<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='DEPTH' type='int'>"<<id.depth()<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='CRATE' type='int'>"<<eid.readoutVMECrateId()<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='SLOT' type='int'>"<<eid.htrSlot()<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='TOPBOTTOM' type='int'>"<<eid.htrTopBottom()<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='FIBER' type='int'>"<<eid.fiberIndex()<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='FIBERCHAN' type='int'>"<<eid.fiberChanId()<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='LUT_TYPE' type='int'>1</Parameter>"<<std::endl;
  os <<" <Parameter name='CREATIONTAG' type='string'>"<<creationtag_<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='CREATIONSTAMP' type='string'>"<<creationstamp_<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='FORMATREVISION' type='string'>"<<formatRevision_<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='TARGETFIRMWARE' type='string'>"<<targetfirmware_<<"</Parameter>"<<std::endl;
  int generalizedIndex=id.ietaAbs()+1000*id.depth()+10000*id.iphi()+
    ((id.ieta()<0)?(0):(100))+((id.subdet()==HcalForward && id.ietaAbs()==29)?(4*10000):(0));

  os <<" <Parameter name='GENERALIZEDINDEX' type='int'>"<<generalizedIndex<<"</Parameter>"<<std::endl;
  // do checksum
  md5_state_t md5er;
  md5_byte_t digest[16];
  unsigned char tool[2];
  md5_init(&md5er);
  for (int i=0; i<128; i++) {
    tool[0]=lut[i]&0xFF;
    tool[1]=(lut[i]>>8)&0xFF;
    md5_append(&md5er,tool,2);
  }
  md5_finish(&md5er,digest);
  os <<" <Parameter name='CHECKSUM' type='string'>";
  for (int i=0; i<16; i++) os << std::hex << (((int)(digest[i]))&0xFF);
  os << "</Parameter>\n";

  *oc_ << "  <Data crate='" << eid.readoutVMECrateId()
       << "' slot='" << eid.htrSlot()
       << "' fpga='" << eid.htrTopBottom()
       << "' fiber='" << eid.fiberIndex()
       << "' fiberchan='" << eid.fiberChanId()
       << "' luttype='1' elements='1' encoding='hex'>";
  for (int i=0; i<16; i++) *oc_ << std::hex << (((int)(digest[i]))&0xFF);    
  *oc_ << "</Data>\n";

  os <<" <Data elements='128' encoding='hex'> "<<std::endl;
  os << std::hex;
  for(int initr2 = 0; initr2 < 128; initr2++){
    os<<lut[initr2]<<" ";
  } 
  os << std::dec;
  os<<std::endl;
  os <<" </Data> "<<std::endl;
  os <<"</CFGBrick> "<<std::endl;
       
}

void
HcalLuttoDB::writeoutlut2(HcalTrigTowerDetId id, HcalElectronicsId eid, const std::vector<unsigned char>& lut, std::ostream& os) {

  os <<"<CFGBrick> "<<std::endl;
  os <<" <Parameter name='IETA' type='int'>"<<id.ieta()<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='IPHI' type='int'>"<<id.iphi()<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='CRATE' type='int'>"<<eid.readoutVMECrateId()<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='SLOT' type='int'>"<<eid.htrSlot()<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='TOPBOTTOM' type='int'>"<<eid.htrTopBottom()<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='SLB' type='int'>"<<eid.slbSiteNumber()<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='SLBCHAN' type='int'>"<<eid.slbChannelIndex()<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='LUT_TYPE' type='int'>2</Parameter>"<<std::endl;
  os <<" <Parameter name='CREATIONTAG' type='string'>"<<creationtag_<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='CREATIONSTAMP' type='string'>"<<creationstamp_<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='FORMATREVISION' type='string'>"<<formatRevision_<<"</Parameter>"<<std::endl;
  os <<" <Parameter name='TARGETFIRMWARE' type='string'>"<<targetfirmware_<<"</Parameter>"<<std::endl;
  int generalizedIndex=id.ietaAbs()+10000*id.iphi()+
    ((id.ieta()<0)?(0):(100));


  os <<" <Parameter name='GENERALIZEDINDEX' type='int'>"<<generalizedIndex<<"</Parameter>"<<std::endl;


// do checksum
  md5_state_t md5er;
  md5_byte_t digest[16];
  md5_init(&md5er);
  md5_append(&md5er,&(lut[0]),1024);
  md5_finish(&md5er,digest);
  os <<" <Parameter name='CHECKSUM' type='string'>";
  for (int i=0; i<16; i++) os << std::hex << (((int)(digest[i]))&0xFF);
  os << "</Parameter>\n";

  *oc_ << "  <Data crate='" << eid.readoutVMECrateId()
       << "' slot='" << eid.htrSlot()
       << "' fpga='" << eid.htrTopBottom()
       << "' slb='" << eid.slbSiteNumber()
       << "' slbchan='" << eid.slbChannelIndex()
       << "' luttype='2' elements='1' encoding='hex'>";
  for (int i=0; i<16; i++) *oc_ << std::hex << (((int)(digest[i]))&0xFF);    
  *oc_ << "</Data>\n";

  os <<" <Data elements='1024' encoding='hex'> "<<std::endl;
  os << std::hex;
  for(int initr2 = 0; initr2 < 1024; initr2++){
    os<< (int(lut[initr2])&0xFF)<<" ";
  } 
  os << std::dec;
  os<<std::endl;
  os <<" </Data> "<<std::endl;
  os <<"</CFGBrick> "<<std::endl;
       
}

std::vector<unsigned char> HcalLuttoDB::extractOutputLut(const CaloTPGTranscoder& coder, HcalTrigTowerDetId chan) {
  std::vector<unsigned char> lut;
  for (int i=0; i<1024; i++) {
    HcalTriggerPrimitiveSample s=coder.hcalCompress(chan,i,false);
    lut.push_back(s.compressedEt());
  }
  return lut;
}

// ------------ method called to produce the data  ------------
void
HcalLuttoDB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //using namespace edm;
  //using namespace std;
  
  edm::LogInfo("Hcal") << "Beginning dump of Hcal TPG LUTS (this may take a minute or two)";

  const HcalElectronicsMap* Map_;
  ESHandle<HcalDbService> pSetup;
  iSetup.get<HcalDbRecord>().get( pSetup );
  Map_=pSetup->getHcalMapping();
  // get the conditions, for the decoding
  edm::ESHandle<HcalTPGCoder> inputCoder;
  iSetup.get<HcalTPGRecord>().get(inputCoder);
  edm::ESHandle<CaloTPGTranscoder> outTranscoder;
  iSetup.get<CaloTPGRecord>().get(outTranscoder);
  outTranscoder->setup(iSetup,CaloTPGTranscoder::HcalTPG);

  std::vector<HcalElectronicsId> allEID = Map_->allElectronicsId();
  std::vector<HcalElectronicsId>::iterator itreid;

  std::ostream* pfile=0;
  oc_=openChecksums();
  
  for (int crate=0; crate<20; crate++) {
    edm::LogInfo("Hcal") << "Beginning crate " << crate;
    for(itreid  = allEID.begin(); itreid != allEID.end(); itreid++)
      {
	if (itreid->readoutVMECrateId()!=crate) continue;
	if (itreid->isTriggerChainId()) { // lut2
	  HcalTrigTowerDetId tid=Map_->lookupTrigger(*itreid);
	  if (tid.null()) continue;

	  if (filePerCrate_ && pfile==0) pfile=openPerCrate(crate);
	  else if (pfile==0) pfile=openPerLut2(*itreid);

	  std::vector<unsigned char> lut=extractOutputLut(*outTranscoder,tid);
	  writeoutlut2(tid,*itreid,lut,*pfile);
	  if (!filePerCrate_) { delete pfile; pfile=0; }	  
	} else { // lut1
	  HcalGenericDetId gid=Map_->lookup(*itreid);
	  if (gid.null() || !(gid.genericSubdet()==HcalGenericDetId::HcalGenBarrel || gid.genericSubdet()==HcalGenericDetId::HcalGenEndcap || gid.genericSubdet()==HcalGenericDetId::HcalGenForward)) continue;
	  
	  if (filePerCrate_ && pfile==0) pfile=openPerCrate(crate);
	  else if (pfile==0) pfile=openPerLut1(*itreid);

	  std::vector<unsigned short> lut=inputCoder->getLinearizationLUT(HcalDetId(gid));
	  writeoutlut1(HcalDetId(gid),*itreid,lut,*pfile);
	  if (!filePerCrate_) { delete pfile; pfile=0; }
	}
      }
    if (pfile!=0) {
      if (filePerCrate_) *pfile << "</CFGBrickSet>\n";
      delete pfile;
      pfile=0;
    }
  }
  *oc_ << "</CFGBrick>\n";
  delete oc_;

  outTranscoder->releaseSetup();
}



// ------------ method called once each job just before starting event loop  ------------
void 
HcalLuttoDB::beginJob()
{
  char buffer[120];
  time_t now=time(0);
  struct tm* tm=localtime(&now);
  strftime(buffer,120,"%F %T",tm);
  creationstamp_     = buffer;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalLuttoDB::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalLuttoDB);
