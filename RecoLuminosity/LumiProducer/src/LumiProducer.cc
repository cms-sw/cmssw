// -*- C++ -*-
//
// Package:    LumiProducer
// Class:      LumiProducer
// 
/**\class LumiProducer LumiProducer.cc RecoLuminosity/LumiProducer/src/LumiProducer.cc

Description: This class would load the luminosity object into a Luminosity Block

Implementation:
The are two main steps, the first one retrieve the record of the luminosity
data from the DB and the second loads the Luminosity Obj into the Lumi Block.
(Actually in the initial implementation it is retrieving from the ParameterSet
from the configuration file, the DB is not implemented yet)
*/
//
// Original Author:  Valerie Halyo
//                   David Dagenhart
//       
//         Created:  Tue Jun 12 00:47:28 CEST 2007
// $Id: LumiProducer.cc,v 1.7 2009/02/20 13:53:41 xiezhen Exp $

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/RunInfo/interface/LuminosityInfo.h"
#include "CondFormats/DataRecord/interface/LuminosityInfoRcd.h"

#include <sstream>
#include <string>
#include <memory>
#include <vector>

namespace edm {
  class EventSetup;
}

//
// class declaration
//

class LumiProducer : public edm::EDProducer {

  public:

    explicit LumiProducer(const edm::ParameterSet&);
    ~LumiProducer();

  private:

    virtual void produce(edm::Event&, const edm::EventSetup&);

    virtual void beginLuminosityBlock(edm::LuminosityBlock & iLBlock,
                                      edm::EventSetup const& iSetup);
    edm::ParameterSet pset_;
};

//
// constructors and destructor
//
LumiProducer::LumiProducer(const edm::ParameterSet& iConfig)
{
  // register your products
  produces<LumiSummary, edm::InLumi>();
  produces<LumiDetails, edm::InLumi>();
  pset_ = iConfig;
}

LumiProducer::~LumiProducer()
{ }
//
// member functions
//

void LumiProducer::produce(edm::Event& e, const edm::EventSetup& iSetup)
{ 
}

void LumiProducer::beginLuminosityBlock(edm::LuminosityBlock &iLBlock, edm::EventSetup const &iSetup) {
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("LuminosityInfoRcd"));
  if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    //record not found
    std::cout <<"Record \"LuminosityInfoRcd"<<"\" does not exist "<<std::endl;
  }
  edm::ESHandle<lumi::LuminosityInfo> pLumi;
  iSetup.get<LuminosityInfoRcd>().get(pLumi);
  const lumi::LuminosityInfo* myLumi=pLumi.product();
  /**summary information
     avginsdellumi: average instante lumi value 
     avginsdellumierr:  average instante lumi error
     lumisecqual: lumi section quality
     lsnumber: lumisection number
     deadfrac: deadtime normalization
     
     l1ratecounter, unavailable now
     hltratecounter, unavailable now
     hltscaler, unavailable now
     hltinput, unavailable now
  */
  float avginsdellumi=myLumi->lumiAverage(lumi::ET).value;
  float avginsdellumierr=myLumi->lumiAverage(lumi::ET).error;
  int lumisecqual=myLumi->lumiAverage(lumi::ET).quality;
  float deadfrac=myLumi->deadTimeNormalization();
  int lsnumber=myLumi->lumisectionID();
  std::vector<int> l1ratecounter;
  for(unsigned int i=0; i<128; ++i){
    l1ratecounter.push_back(-99);
  }
  std::vector<int> l1scaler;
  for(unsigned int i=0; i<128; ++i){
    l1scaler.push_back(-99);
  }
  std::vector<int> hltratecounter;
  for(unsigned int i=0; i<128; ++i){
    hltratecounter.push_back(-99);
  }
  std::vector<int> hltscaler;
  for(unsigned int i=0; i<128; ++i){
    hltscaler.push_back(-99);
  }
  std::vector<int> hltinput;
  for(unsigned int i=0; i<128; ++i){
    hltinput.push_back(-99);
  }
  LumiSummary* pIn1=new LumiSummary(avginsdellumi,avginsdellumierr,lumisecqual,deadfrac,lsnumber,l1ratecounter,l1scaler,hltratecounter,hltscaler,hltinput);
  std::auto_ptr<LumiSummary> pOut1(pIn1);
  iLBlock.put(pOut1);
  
  /**detailed information for all bunchcrossings
     lumietsum: lumi et values 
     lumietsumerr: lumi et errors
     lumietsumqual: lumi et qualities

     lumiocc, lumi occ values
     lumioccerr, lumi occ errors
     lumioccerr, lumi occ qualities
  */
  std::vector<lumi::BunchCrossingInfo> resultET;
  myLumi->bunchCrossingInfo(lumi::ET,resultET);
  std::vector<float> lumietsum;
  std::vector<float> lumietsumerr;
  std::vector<int> lumietsumqual;
  for(std::vector<lumi::BunchCrossingInfo>::iterator it=resultET.begin();
      it!=resultET.end();++it){
    lumietsum.push_back(it->lumivalue);
    lumietsumerr.push_back(it->lumierr);
    lumietsumqual.push_back(it->lumiquality);
  }
  std::vector<lumi::BunchCrossingInfo> resultOCCD1;
  myLumi->bunchCrossingInfo(lumi::OCCD1,resultOCCD1);
  std::vector<float> lumiocc;
  std::vector<float> lumioccerr;
  std::vector<int> lumioccqual;
  for(std::vector<lumi::BunchCrossingInfo>::iterator it=resultOCCD1.begin();
      it!=resultOCCD1.end();++it){
    lumiocc.push_back(it->lumivalue);
    lumioccerr.push_back(it->lumierr);
    lumioccqual.push_back(it->lumiquality);
  }
  LumiDetails* pIn2=new LumiDetails(lumietsum,lumietsumerr,lumietsumqual,lumiocc, lumioccerr,lumioccqual);
  std::auto_ptr<LumiDetails> pOut2(pIn2);
  iLBlock.put(pOut2);
}

DEFINE_FWK_MODULE(LumiProducer);
