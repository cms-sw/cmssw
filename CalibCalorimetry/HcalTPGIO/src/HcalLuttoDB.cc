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
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/CaloTPG/interface/HcalTPGCompressor.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
using namespace edm;
using namespace std;
#include <iostream>
#include <fstream>

//
// class decleration
//

class HcalLuttoDB : public edm::EDProducer {
public:
  explicit HcalLuttoDB(const edm::ParameterSet&);
  ~HcalLuttoDB();
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  void writeoutlut(int, int, int, int, int, int, int, int, int[], int[] );
  virtual void endJob() ;


};      
      // ----------member data ---------------------------
 
std::string creationtag_;
std::string creationstamp_;
std::string formatrevision_;
std::string targetfirmware_;
std::string generalizedindex_;
ofstream myfile;
int fakein[128] = {0};
int fakeout[1024] = {0};
//
// constructors and destructor
//
HcalLuttoDB::HcalLuttoDB(const edm::ParameterSet& iConfig)
{
  
  creationtag_       = iConfig.getParameter<std::string>("creationtag");
  creationstamp_     = iConfig.getParameter<std::string>("creationstamp");
  formatrevision_    = iConfig.getParameter<std::string>("formatrevision");
  targetfirmware_    = iConfig.getParameter<std::string>("targetfirmware");
  generalizedindex_  = iConfig.getParameter<std::string>("generalizedindex");
  
    //register your products
  produces<int>();
}


HcalLuttoDB::~HcalLuttoDB()
{
}


//
// member functions
//

void
HcalLuttoDB::writeoutlut(int ieta_, int iphi_, int depth_, int crate_, int slot_,
			 int topbottom_, int fiber_, int fiberchan_,
			 int inputlut_[], int outputlut_[])
{
  //using namespace edm;
  //using namespace std;
  
  int ieta = ieta_;
  int iphi = iphi_;
  int depth = depth_;
  int crate = crate_;
  int slot = slot_;
  int topbottom = topbottom_;
  int fiber = fiber_;
  int fiberchan = fiberchan_;
  const int inputsize = 128;
  const int outputsize = 1024;
  // int inputlut[] = inputlut_;
  //int outputlut[] = outputlut_;
  //  ofstream myfile;
  myfile <<"<pre> "<<std::endl;
  myfile <<"<CFGBrick> "<<std::endl;
  myfile <<" <Parameter name='IETA' type='int'>"<<ieta<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='IPHI' type='int'>"<<iphi<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='DEPTH' type='int'>"<<depth<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='CRATE' type='int'>"<<crate<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='SLOT' type='int'>"<<slot<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='TOPBOTTOM' type='int'>"<<topbottom<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='FIBER' type='int'>"<<fiber<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='FIBERCHAN' type='int'>"<<fiberchan<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='LUT_TYPE' type='int'>1</Parameter>"<<std::endl;
  myfile <<" <Parameter name='CREATIONTAG' type='string'>"<<creationtag_<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='CREATIONSTAMP' type='string'>"<<creationstamp_<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='FORMATREVISION' type='string'>"<<formatrevision_<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='TARGETFIRMWARE' type='string'>"<<targetfirmware_<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='GENERALIZEDINDEX' type='int'>"<<generalizedindex_<<"</Parameter>"<<std::endl;
  myfile <<" <Data elements='128' encoding='hex'> "<<std::endl;
  myfile << std::hex;
  for(int initr2 = 0; initr2 < inputsize; initr2++){
    myfile<<inputlut_[initr2]<<" ";
  } 
  myfile << std::dec;
  myfile<<std::endl;
  myfile <<" </Data> "<<std::endl;
  myfile <<"</CFGBrick> "<<std::endl;
  myfile <<"</pre> "<<std::endl;
  
  myfile <<"<pre> "<<std::endl;
  myfile <<"<CFGBrick> "<<std::endl;
  myfile <<" <Parameter name='IETA' type='int'>"<<ieta<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='IPHI' type='int'>"<<iphi<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='DEPTH' type='int'>"<<depth<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='CRATE' type='int'>"<<crate<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='SLOT' type='int'>"<<slot<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='TOPBOTTOM' type='int'>"<<topbottom<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='FIBER' type='int'>"<<fiber<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='FIBERCHAN' type='int'>"<<fiberchan<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='LUT_TYPE' type='int'>2</Parameter>"<<std::endl;
  myfile <<" <Parameter name='CREATIONTAG' type='string'>"<<creationtag_<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='CREATIONSTAMP' type='string'>"<<creationstamp_<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='FORMATREVISION' type='string'>"<<formatrevision_<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='TARGETFIRMWARE' type='string'>"<<targetfirmware_<<"</Parameter>"<<std::endl;
  myfile <<" <Parameter name='GENERALIZEDINDEX' type='int'>"<<generalizedindex_<<"</Parameter>"<<std::endl;
  myfile <<" <Data elements='1024' encoding='hex'> "<<std::endl;
  myfile << std::hex;
  for(int outitr2 = 0; outitr2 < outputsize; outitr2++){
    myfile<<outputlut_[outitr2]<<" ";
  } 
  myfile << std::dec;
  myfile<<std::endl;
  myfile <<" </Data> "<<std::endl;
  myfile <<"</CFGBrick> "<<std::endl;
  myfile <<"</pre> "<<std::endl;
       
}


// ------------ method called to produce the data  ------------
void
HcalLuttoDB::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //using namespace edm;
  //using namespace std;
  
  
  int numIDs = 0;
  int ieta_;
  int iphi_;
  int depth_;
  int crate_ = 666;
  int slot_;
  int topbottom_;
  int fiber_;
  int fiberchan_;
  int current_crate_;
  
  
  const int inputsize_ = 128;
  const int outputsize_ = 1024;
  int inputlut_[inputsize_];
  int outputlut_[outputsize_];
  
  bool parts_used[21][2][9][3] = {{{{false}}}}; //[htr][TB][fiber][chan]
  bool htr_used[22] = {false};
  //const HcalElectronicsMap* Map_;
  HcalElectronicsMap* Map_;
  ESHandle<HcalDbService> pSetup;
  iSetup.get<HcalDbRecord>().get( pSetup );
  Map_=pSetup->getHcalMapping();
  
  //  Map_->sortByElectronicsId();
  std::vector <HcalElectronicsId> allEID = Map_->allElectronicsId();
  std::vector<HcalElectronicsId>::iterator itrseid2;  
  // for(uint32_t counter_ = 0; counter_ < 2^32; counter_++)
  /* for(itrseid2  = allEID.begin(); itrseid2 < allEID.end(); itrseid2++)
    {
      std::cout<<itrseid2->readoutVMECrateId()<<" "<<itrseid2->htrSlot()<<" "<<
	itrseid2->htrTopBottom()<<" "<<itrseid2->fiberIndex()<<" "<<itrseid2->fiberChanId()<<std::endl;
	}*/

  Map_->sortByElectronicsId();
  std::vector <HcalElectronicsId> sortedEID = Map_->allElectronicsId();
  // get the conditions, for the decoding
  edm::ESHandle<HcalTPGCoder> inputCoder;
  iSetup.get<HcalTPGRecord>().get(inputCoder);
  inputCoder->getConditions(iSetup);
  
  current_crate_ = sortedEID.begin()->readoutVMECrateId();
  std::stringstream ss;
  std::string filename;
  ss << "Luts_"<<current_crate_<<".xml";
  ss >> filename;
  
  
  std::cout<<filename<<std::endl;
  myfile.open(filename.c_str());
  
  // edm::ESHandle<CaloTPGTranscoder> outTranscoder;
  //iSetup.get<CaloTPGRecord>().get(outTranscoder);
  //outTranscoder->setup(iSetup,CaloTPGTranscoder::HcalTPG);
  
  std::vector<HcalElectronicsId>::iterator itrseid;  
  // for(uint32_t counter_ = 0; counter_ < 2^32; counter_++)
  for(itrseid  =sortedEID.begin(); itrseid < sortedEID.end(); itrseid++)
    {
      // HcalElectronicsId EID_(counter_);
      //HcalDetId detid_(Map_->lookup(EID_));
      
      // std::cout<<"Number of channels in loop "<<numIDs<<std::endl;
      if(Map_->lookup(*itrseid).null() ||
	 Map_->lookup(*itrseid).subdetId() == HcalOuter)
	{
	  // std::cout<<"Null or HO "<<itrseid2->readoutVMECrateId()<<" "<<itrseid2->htrSlot()<<" "<<
	  //itrseid2->htrTopBottom()<<" "<<itrseid2->fiberIndex()<<" "<<itrseid2->fiberChanId()<<std::endl;
	  continue;}
      HcalSubdetector subdet=(HcalSubdetector(Map_->lookup(*itrseid).subdetId()));
      if (Map_->lookup(*itrseid).det()!= 4 || 
	  (subdet!=HcalBarrel && subdet!=HcalEndcap && 
	   subdet!=HcalOuter && subdet!=HcalForward ))
	{
	  //std::cout<<"Not Hcal or HB-HE-HO-HF "<<itrseid2->readoutVMECrateId()<<" "<<itrseid2->htrSlot()<<" "<<
	  //itrseid2->htrTopBottom()<<" "<<itrseid2->fiberIndex()<<" "<<itrseid2->fiberChanId()<<std::endl;
	  continue;}
      
      // std::cout<<"EID CODE: "<<itrseid<<std::endl;
      HcalDetId detid_(Map_->lookup(*itrseid));
      
      numIDs++;
      ieta_      = detid_.ieta();
      iphi_      = detid_.iphi();
      depth_     = detid_.depth();
      crate_     = itrseid->readoutVMECrateId();
      slot_      = itrseid->htrSlot();
      topbottom_ = itrseid->htrTopBottom();
      fiber_     = itrseid->fiberIndex();
      fiberchan_ = itrseid->fiberChanId();
      
      parts_used[slot_][topbottom_][fiber_][fiberchan_] = true;
      htr_used[slot_] = true;
      for(int initr = 0; initr < inputsize_; initr++)
	{
	  HBHEDataFrame frame(detid_);
	  frame.setSize(1);
	  HcalQIESample qie(initr, 1, fiber_, fiberchan_, false);
	  frame.setSample(0, qie);	   IntegerCaloSamples samples(detid_, 1);
	  inputCoder.product()->adc2Linear(frame, samples);
	  inputlut_[initr] = samples[0];   
	}
      
      for(int outitr = 0; outitr < outputsize_; outitr++)
	{
	  /*IntegerCaloSamples output(detid_,1);
	    output.setPresamples(0);
	    output[0] = outitr;
	    std::vector<bool> finegrain(1,false);
	    HcalTriggerPrimitiveDigi result;	   
	    outTranscoder->getHcalCompressor().get()->compress(output, finegrain, result);
	  */
	  outputlut_[outitr] = outitr;//result.SOI_compressedEt();
	}       
      
      //Create XML file for channel
      
      
      if(current_crate_ != crate_)
	{
	  for(int i = 0; i<21; i++){
	    for(int j = 0; j<2; j++){
	      for(int k = 1; k<9; k++){
		for(int l = 0; l<3; l++){
		  if(htr_used[i] && parts_used[i][j][k][l] == false)
		    {
		      writeoutlut(0, 0, 0, crate_, i, j, k, l, fakein, fakeout);
		      std::cout<<"Crate: "<<crate_<<" Slot "<<i<<" TB "<<j<<" Fiber "<<k<<" Chan "<<l<<" FAKE "<<std::endl;
		    }
		}}}}
	  for(int i = 0; i<21; i++){
	    for(int j = 0; j<2; j++){
	      for(int k = 0; k<9; k++){
		for(int l = 0; l<3; l++){
		  htr_used[i] = false;
		  parts_used[i][j][k][l] = false;
		}}}}
	  current_crate_ = crate_; 
	  myfile.close();
	  std::stringstream ss;
	  std::string filename;
	  
	  ss << "Luts_"<<current_crate_<<".xml";
	  ss >> filename;
	  std::cout<<filename<<std::endl;
	  myfile.open(filename.c_str());
	  
	  
	}
      //Write out info to XML file

      std::cout<<"Crate: "<<crate_<<" Slot "<< slot_<<" TB "<<topbottom_<<" Fiber "<<fiber_<<" Chan "<< fiberchan_<<std::endl;
      writeoutlut(ieta_, iphi_, depth_, crate_, slot_, topbottom_, fiber_, fiberchan_, inputlut_, outputlut_);
      //writeoutlut(3, 3, 3, 3, 3, 3, 3, 3, inputlut_[], outputlut_[]);
      
    }
  std::cout<<"Number of channels "<<numIDs<<std::endl;
  for(int i = 0; i<21; i++){
    for(int j = 0; j<2; j++){
      for(int k = 1; k<9; k++){
	for(int l = 0; l<3; l++){
	  if(htr_used[i] && parts_used[i][j][k][l] == false)
	    {
	      writeoutlut(0, 0, 0, crate_, i, j, k, l, fakein, fakeout);
	      std::cout<<"Crate: "<<crate_<<" Slot "<<i<<" TB "<<j<<" Fiber "<<k<<" Chan "<<l<<" FAKE"<<std::endl;
	    }
	}}}}
  myfile.close();
}



// ------------ method called once each job just before starting event loop  ------------
void 
HcalLuttoDB::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalLuttoDB::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalLuttoDB);
