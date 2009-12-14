// -*- C++ -*-
//
// Package:    TestPerformanceFW_ES
// Class:      TestPerformanceFW_ES
// 
/**\class TestPerformanceFW_ES TestPerformanceFW_ES.cc RecoBTag/TestPerformanceFW_ES/src/TestPerformanceFW_ES.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tommaso Boccali
//         Created:  Tue Nov 25 15:50:50 CET 2008
// $Id: TestPerformanceFW_ES.cc,v 1.2 2009/08/13 12:33:24 tboccali Exp $
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

class TestPerformanceFW_ES : public edm::EDAnalyzer {
public:
  explicit TestPerformanceFW_ES(const edm::ParameterSet&);
  ~TestPerformanceFW_ES();
  
  
private:
  std::string name;
  virtual void beginJob() ;
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
TestPerformanceFW_ES::TestPerformanceFW_ES(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
  std::cout <<" In the constructor"<<std::endl;
  
  name =  iConfig.getParameter<std::string>("AlgoName");
}


TestPerformanceFW_ES::~TestPerformanceFW_ES()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TestPerformanceFW_ES::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::ESHandle<BtagPerformance> perfH;
  std::cout <<" Studying performance with label "<<name <<std::endl;
  iSetup.get<BTagPerformanceRecord>().get(name,perfH);

  const BtagPerformance & perf = *(perfH.product());


  std::cout << "Values: "<<
    PerformanceResult::BTAGNBEFF<<" " <<
    PerformanceResult::MUERR<<" " <<
    std::endl;

  // check beff, berr for eta=.6, et=55;
  BinningPointByMap p;

  std::cout <<" My Performance Object is indeed a "<<typeid(perfH.product()).name()<<std::endl;

  std::cout <<" test eta=0.6, et=55"<<std::endl;


  p.insert(BinningVariables::JetEta,0.6);
  p.insert(BinningVariables::JetEt,55);
  std::cout <<" nbeff/nberr ?"<<perf.isResultOk(PerformanceResult::BTAGNBEFF,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGNBERR,p)<<std::endl;
  std::cout <<" beff/berr ?"<<perf.isResultOk(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGBERR,p)<<std::endl;
  std::cout <<" beff/berr ="<<perf.getResult(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.getResult(PerformanceResult::BTAGBERR,p)<<std::endl;

  std::cout <<" test eta=1.9, et=33"<<std::endl;
   p.insert(BinningVariables::JetEta,1.9);
  p.insert(BinningVariables::JetEt,33);
  std::cout <<" beff/berr ?"<<perf.isResultOk(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGBERR,p)<<std::endl;
  std::cout <<" beff/berr ="<<perf.getResult(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.getResult(PerformanceResult::BTAGBERR,p)<<std::endl;

  std::cout <<" The WP is defined by a cut at "<<perf.workingPoint().cut()<<std::endl;
  std::cout <<" Discriminant is "<<perf.workingPoint().discriminantName()<<std::endl;

  std::cout <<" now I ask for a calibration but I do not set eta in the binning point ---> should return all not available "<<std::endl;
  p.reset();
  p.insert(BinningVariables::JetNTracks,3);
  p.insert(BinningVariables::JetEt,55);
  std::cout <<" beff/berr ?"<<perf.isResultOk(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGBERR,p)<<std::endl;
  std::cout <<" beff/berr ="<<perf.getResult(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.getResult(PerformanceResult::BTAGBERR,p)<<std::endl;

  //  std::cout <<" now I ask for a calibration which is not present ---> should throw an exception "<<std::endl;

  //  edm::ESHandle<BtagPerformance> perfH2;
  //  iSetup.get<BTagPerformanceRecord>().get("TrackCountingHighEff_tight",perfH2);
  

}


// ------------ method called once each job just before starting event loop  ------------
void 
TestPerformanceFW_ES::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TestPerformanceFW_ES::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestPerformanceFW_ES);
