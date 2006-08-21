using namespace std;

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBRunData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBEventPosition.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBBeamCounters.h"
#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalTBObjectUnpacker.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"
#include <iostream>
#include <fstream>


  HcalTBObjectUnpacker::HcalTBObjectUnpacker(edm::ParameterSet const& conf):
    triggerFed_(conf.getUntrackedParameter<int>("HcalTriggerFED",-1)),
    sdFed_(conf.getUntrackedParameter<int>("HcalSlowDataFED",-1)),
    spdFed_(conf.getUntrackedParameter<int>("HcalSourcePositionFED",-1)),
    tdcFed_(conf.getUntrackedParameter<int>("HcalTDCFED",-1)),
    qadcFed_(conf.getUntrackedParameter<int>("HcalQADCFED",-1)),
    calibFile_(conf.getUntrackedParameter<string>("ConfigurationFile","")),
    tdcUnpacker_(conf.getUntrackedParameter<bool>("IncludeUnmatchedHits",false)),
    doRunData_(false),doTriggerData_(false),doEventPosition_(false),doTiming_(false),doSourcePos_(false),doBeamADC_(false)
  {

    calibLines_.clear();
    if(calibFile_.size()>0){
      parseCalib();
    //  printf("I got %d lines!\n",calibLines_.size());
    if(calibLines_.size()==0)
	throw cms::Exception("Incomplete configuration") << 
	  "HcalTBObjectUnpacker: TDC/QADC/WC configuration file not found or is empty: "<<calibFile_<<endl;
    }
    else{
	throw cms::Exception("Incomplete configuration") << 
	  "HcalTBObjectUnpacker: TDC/QADC/WC configuration file not found: "<<calibFile_<<endl;
    }

    if (triggerFed_ >=0) {
      std::cout << "HcalTBObjectUnpacker will unpack FED ";
      std::cout << triggerFed_ << endl;
      doTriggerData_=true;
    }

    if (sdFed_ >=0) {
      std::cout << "HcalTBObjectUnpacker will unpack SlowData FED ";
      std::cout << sdFed_ << endl;
      doRunData_=true;
      doEventPosition_=true; // at least the table
    }

    if (tdcFed_ >=0) {
      std::cout << "HcalTBObjectUnpacker will unpack TDC FED ";
      std::cout << tdcFed_ << endl;
      doTiming_=true;
      doEventPosition_=true; // at least the WC
    }

    if (qadcFed_ >=0) {
      std::cout << "HcalTBObjectUnpacker will unpack QADC FED ";
      std::cout << qadcFed_ << endl;
      doBeamADC_=true;
    }

    if (spdFed_ >=0) {
      std::cout << "HcalTBObjectUnpacker will unpack Source Position Data FED ";
      std::cout << spdFed_ << endl;
      doSourcePos_=true;
    }

    if (doTriggerData_) produces<HcalTBTriggerData>();
    if (doRunData_) produces<HcalTBRunData>();
    if (doEventPosition_) produces<HcalTBEventPosition>();
    if (doTiming_) produces<HcalTBTiming>();
//    if (doBeamADC_) produces<HcalTBBeamCounters>();
    if (doBeamADC_) {produces<HcalTBBeamCounters>();qadcUnpacker_.setCalib(calibLines_);}
//    if (doSourcePos_) produces<HcalSourcePositionData>();
    if(doTiming_||doEventPosition_)tdcUnpacker_.setCalib(calibLines_);
  }

  // Virtual destructor needed.
  HcalTBObjectUnpacker::~HcalTBObjectUnpacker() { }  

  // Functions that gets called by framework every event
  void HcalTBObjectUnpacker::produce(edm::Event& e, const edm::EventSetup&)
  {
    // Step A: Get Inputs 
    edm::Handle<FEDRawDataCollection> rawraw;  
    //    edm::ProcessNameSelector s("PROD"); // HACK!
    e.getByType(rawraw);           

    // Step B: Create empty output    
    std::auto_ptr<HcalTBTriggerData>
      trigd(new HcalTBTriggerData);

    std::auto_ptr<HcalTBRunData>
      rund(new HcalTBRunData);

    std::auto_ptr<HcalTBEventPosition>
      epd(new HcalTBEventPosition);

    std::auto_ptr<HcalTBTiming>
      tmgd(new HcalTBTiming);

    std::auto_ptr<HcalTBBeamCounters>
      bcntd(new HcalTBBeamCounters);

    std::auto_ptr<HcalSourcePositionData>
      spd(new HcalSourcePositionData);
    
    if (triggerFed_ >=0) {
      // Step C: unpack all requested FEDs
      const FEDRawData& fed = rawraw->FEDData(triggerFed_);
      tdUnpacker_.unpack(fed,*trigd);
    }

    if (sdFed_ >=0) {
      // Step C: unpack all requested FEDs
      const FEDRawData& fed = rawraw->FEDData(sdFed_);
      sdUnpacker_.unpack(fed, *rund, *epd);
    }

    if (tdcFed_ >=0) {
      // Step C: unpack all requested FEDs
      const FEDRawData& fed = rawraw->FEDData(tdcFed_);
      tdcUnpacker_.unpack(fed, *epd, *tmgd);
    }

    if (qadcFed_ >=0) {
      // Step C: unpack all requested FEDs
      const FEDRawData& fed = rawraw->FEDData(qadcFed_);
      bool is04 = true;
      if(qadcFed_==8) is04=false;
      qadcUnpacker_.unpack(fed, *bcntd,is04);
    }

    if (spdFed_ >=0) {      
      // Step C: unpack all requested FEDs
      const FEDRawData& fed = rawraw->FEDData(spdFed_);
      spdUnpacker_.unpack(fed, *spd);
    }

    // Step D: Put outputs into event
    if (doTriggerData_) e.put(trigd);
    if (doRunData_) e.put(rund);
    if (doEventPosition_) e.put(epd);
    if (doTiming_) e.put(tmgd);
    if (doBeamADC_) e.put(bcntd);
    if (doSourcePos_) e.put(spd);
  }


void HcalTBObjectUnpacker::parseCalib(){
  
  if(calibFile_.size()==0){
    printf("HcalTBObjectUnpacker cowardly refuses to parse a NULL file...\n"); 
    return;
  }
 
  ifstream infile(calibFile_.c_str());

  char buffer [1024];
  string tmpStr;
  
  while (infile.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    if (buffer [0] == '/' && buffer [1] == '/') continue; //ignore comment
    tmpStr = string(buffer);
    vector<string> lineVect;
    
    int start = 0; bool empty = true;
    for (unsigned i=0; i<=tmpStr.size(); i++) {
      if (tmpStr[i] == ' ' || i==tmpStr.size()) {
	if (!empty) {
	  std::string item(tmpStr, start, i-start);
	  lineVect.push_back(item);
	  empty = true;
//	  printf("Got: %s\n",item.c_str());
	}
	start = i+1;
      }
      else {
	if (empty) empty = false;	
      }
    }  
    
    if(lineVect.size()>0) calibLines_.push_back(lineVect);    
  }

  infile.close();
  return;
}
