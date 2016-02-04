// -*- C++ -*-
//
// Package:    TestOctoberExe
// Class:      TestOctoberExe
// 
/**\class TestOctoberExe TestOctoberExe.cc RecoBTag/TestOctoberExe/src/TestOctoberExe.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tommaso Boccali
//         Created:  Tue Nov 25 15:50:50 CET 2008
// $Id: TestOctober_ES.cc,v 1.1 2010/01/19 11:14:48 tboccali Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
//
// class decleration
//

#include "RecoBTag/Records/interface/BTagPerformanceRecord.h"

#include "CondFormats/PhysicsToolsObjects/interface/BinningPointByMap.h"
#include "RecoBTag/PerformanceDB/interface/BtagPerformance.h"
//#include "CondFormats/BTagPerformance/interface/BtagPerformancePayloadFromTableEtaJetEt.h"
//#include "CondFormats/BTagPerformance/interface/BtagPerformancePayloadFromTableEtaJetEtPhi.h"

class TestOctoberExe : public edm::EDAnalyzer {
public:
  explicit TestOctoberExe(const edm::ParameterSet&);
  ~TestOctoberExe();
  
  
private:
  std::string beff,mistag;
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------
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
TestOctoberExe::TestOctoberExe(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
  std::cout <<" In the constructor"<<std::endl;
  
  beff =  iConfig.getParameter<std::string>("CalibrationForBEfficiency");
  mistag =  iConfig.getParameter<std::string>("CalibrationForMistag");
}


TestOctoberExe::~TestOctoberExe()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TestOctoberExe::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::ESHandle<BtagPerformance> perfH;
  edm::ESHandle<BtagPerformance> perfH1;
  std::cout <<" Studying Beff with label "<<beff <<std::endl;
  std::cout <<" Studying Mistag with label "<<mistag <<std::endl;
  iSetup.get<BTagPerformanceRecord>().get(mistag,perfH1);
  iSetup.get<BTagPerformanceRecord>().get(beff,perfH);

  const BtagPerformance & pbeff = *(perfH.product());
  const BtagPerformance & pmistag = *(perfH1.product());

  // check beff, berr for eta=.6, et=55;
  BinningPointByMap p;

  std::cout <<" My Performance Objects are indeed  "<<typeid(perfH.product()).name()<<" " << typeid(perfH1.product()).name() <<std::endl;



  std::cout <<" test eta=0.6, et=55"<<std::endl;


  p.insert(BinningVariables::JetAbsEta,0.6);
  p.insert(BinningVariables::JetEt,55);

  std::cout <<" mistag/mistagerr ?"<<pmistag.isResultOk(PerformanceResult::BTAGLEFF,p)<<"/"<<pmistag.isResultOk(PerformanceResult::BTAGLERR,p)<<std::endl;
  std::cout <<" beff/berr ?"<<pbeff.isResultOk(PerformanceResult::BTAGBEFF,p)<<"/"<<pbeff.isResultOk(PerformanceResult::BTAGBERR,p)<<std::endl;
  std::cout <<" beffcorr/berrcorr ?"<<pbeff.isResultOk(PerformanceResult::BTAGBEFFCORR,p)<<"/"<<pbeff.isResultOk(PerformanceResult::BTAGBERRCORR,p)<<std::endl;
  std::cout <<" mistag/mistagerr ?"<<pmistag.getResult(PerformanceResult::BTAGLEFF,p)<<"/"<<pmistag.getResult(PerformanceResult::BTAGLERR,p)<<std::endl;
  std::cout <<" beff/berr ?"<<pbeff.getResult(PerformanceResult::BTAGBEFF,p)<<"/"<<pbeff.getResult(PerformanceResult::BTAGBERR,p)<<std::endl;
  std::cout <<" beffcorr/berrcorr ?"<<pbeff.getResult(PerformanceResult::BTAGBEFFCORR,p)<<"/"<<pbeff.getResult(PerformanceResult::BTAGBERRCORR,p)<<std::endl;

}


// ------------ method called once each job just before starting event loop  ------------
void 
TestOctoberExe::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TestOctoberExe::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestOctoberExe);
