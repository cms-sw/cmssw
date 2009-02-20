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
// $Id: LumiProducer.cc,v 1.5 2009/02/20 09:14:05 xiezhen Exp $

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
  std::cout<<"produce"<<std::endl;
  
}

void LumiProducer::beginLuminosityBlock(edm::LuminosityBlock &iLBlock, edm::EventSetup const &iSetup) {
  std::cout<<"beginLuminosityBlock"<<std::endl;
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("LuminosityInfoRcd"));
  if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    //record not found
    std::cout <<"Record \"LuminosityInfoRcd"<<"\" does not exist "<<std::endl;
  }
  edm::ESHandle<lumi::LuminosityInfo> pLumi;
  std::cout<<"got eshandle"<<std::endl;
  iSetup.get<LuminosityInfoRcd>().get(pLumi);
  const lumi::LuminosityInfo* myLumi=pLumi.product();
  std::cout<<" payload pointer "<<myLumi<<std::endl;
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
  std::vector<int> l1scaler;
  std::vector<int> hltratecounter;
  std::vector<int> hltscaler;
  std::vector<int> hltinput;

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
  std::vector<float> lumietsum;
  std::vector<float> lumietsumerr;
  std::vector<int> lumietsumqual;

  std::vector<float> lumiocc;
  std::vector<float> lumioccerr;
  std::vector<int> lumioccqual;
  LumiDetails* pIn2=new LumiDetails(lumietsum,lumietsumerr,lumietsumqual,lumiocc, lumioccerr,lumioccqual);
  std::auto_ptr<LumiDetails> pOut2(pIn2);
  iLBlock.put(pOut2);
}

DEFINE_FWK_MODULE(LumiProducer);
