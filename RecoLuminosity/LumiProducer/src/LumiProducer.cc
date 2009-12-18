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
// $Id: LumiProducer.cc,v 1.14 2009/11/13 14:05:22 xiezhen Exp $

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
#include "FWCore/Framework/interface/NoRecordException.h"
#include "CondFormats/Luminosity/interface/LumiSectionData.h"
#include "CondFormats/DataRecord/interface/LumiSectionDataRcd.h"
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
  void fillDefaultLumi(edm::LuminosityBlock & iLBlock);

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

LumiProducer::~LumiProducer(){ 
}
//
// member functions
//

void LumiProducer::produce(edm::Event& e, const edm::EventSetup& iSetup){ 
}
void LumiProducer::fillDefaultLumi(edm::LuminosityBlock &iLBlock){
  LumiSummary* pIn1=new LumiSummary;
  std::auto_ptr<LumiSummary> pOut1(pIn1);
  iLBlock.put(pOut1);
  LumiDetails* pIn2=new LumiDetails;
  std::auto_ptr<LumiDetails> pOut2(pIn2);
  iLBlock.put(pOut2);
}
void LumiProducer::beginLuminosityBlock(edm::LuminosityBlock &iLBlock, edm::EventSetup const &iSetup) {
  //edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("LumiSectionDataRcd"));
  //if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
  //record not found
  //std::cout <<"Record \"LumiSectionDataRcd"<<"\" does not exist "<<std::endl;
  //}
  try{
    const lumi::LumiSectionData* myLumi=0;
    edm::ESHandle<lumi::LumiSectionData> pLumi;
    iSetup.get<LumiSectionDataRcd>().get(pLumi);
    myLumi=pLumi.product();
    if(!myLumi){
      //std::cout<<"filling default because no lumi data found"<<std::endl;
      //std::string errmsg("NULL lumi object ");
      this->fillDefaultLumi(iLBlock);
      return;
      //throw cms::Exception(" LumiProducer",errmsg);
    }
    if(myLumi->lumiVersion()=="-99"){
      //std::cout<<"filling default because no lumi version -99"<<std::endl;
      this->fillDefaultLumi(iLBlock);
      return;
    }
    /**summary information
       if avginsdellumi is -99, it signals that there is no lumi data written for this lumisection,consequently, all the l1 and hlt values are empty. So users should check and decide what to do.
       
       avginsdellumi: average instante lumi value 
       avginsdellumierr:  average instante lumi error
       lumisecqual: lumi section quality
       lsnumber: lumisection number
       deadfrac: deadtime normalization
       l1data
       hldata
    */
    float avginsdellumi=myLumi->lumiAverage();
    float avginsdellumierr=myLumi->lumiError();
    short lumisecqual=myLumi->lumiquality();
    float deadfrac=myLumi->deadFraction();
    int lsnumber=myLumi->lumisectionID();
    unsigned long long startOrbit=myLumi->startorbit();
    std::vector<LumiSummary::L1> l1data;
    for(lumi::TriggerIterator it=myLumi->trgBegin(); it!=myLumi->trgEnd(); ++it){
      LumiSummary::L1 l;
      l.triggersource=it->name;
      l.ratecount=it->triggercount;
      l.deadtimecount=it->deadtimecount;
      l.scalingfactor=it->prescale;
      l1data.push_back( l );
    }
    
    std::vector<LumiSummary::HLT> hltdata;
    for(lumi::HLTIterator it=myLumi->hltBegin(); it!=myLumi->hltEnd();++it){
      LumiSummary::HLT h;
      h.pathname=it->pathname;
      h.ratecount=it->acceptcount;
      h.inputcount=it->inputcount;
      h.scalingfactor=it->prescale;
      hltdata.push_back(h);
    }
    
    LumiSummary* pIn1=new LumiSummary(avginsdellumi,avginsdellumierr,lumisecqual,deadfrac,lsnumber,l1data,hltdata,startOrbit);
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
  }catch(const edm::eventsetup::NoRecordException<LumiSectionDataRcd>& er){
    //std::cout<<"filling default because NoRecordException"<<std::endl;
    this->fillDefaultLumi(iLBlock);
    return;
  }
}

DEFINE_FWK_MODULE(LumiProducer);
