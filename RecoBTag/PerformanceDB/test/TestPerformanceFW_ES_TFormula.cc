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
//
//

// system include files
#include <iostream>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

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

class TestPerformanceFW_ES_TFormula : public edm::one::EDAnalyzer<> {
public:
  explicit TestPerformanceFW_ES_TFormula(const edm::ParameterSet&);

private:
  std::string name_;
  edm::ESGetToken<BtagPerformance, BTagPerformanceRecord> token_;
  void analyze(const edm::Event&, const edm::EventSetup&) final;

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

TestPerformanceFW_ES_TFormula::TestPerformanceFW_ES_TFormula(const edm::ParameterSet& iConfig)

{
  //now do what ever initialization is needed
  std::cout << " In the constructor" << std::endl;

  name_ = iConfig.getParameter<std::string>("AlgoName");

  token_ = esConsumes(edm::ESInputTag("", name_));
}

//
// member functions
//

// ------------ method called to for each event  ------------
void TestPerformanceFW_ES_TFormula::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::cout << " Studying performance with label " << name_ << std::endl;

  const BtagPerformance& perf = iSetup.getData(token_);

  //std::cout << "Values: "<<
  //  PerformanceResult::BTAGNBEFF<<" " <<
  //  PerformanceResult::MUERR<<" " <<
  //  std::endl;

  // check beff, berr for eta=.6, et=55;
  BinningPointByMap p;

  std::cout << " My Performance Object is indeed a " << typeid(&perf).name() << std::endl;

  //++++++++++++------  TESTING FOR CONTINIOUS DISCRIMINATORS      --------+++++++++++++

  std::cout << " The WP is defined by a cut at " << perf.workingPoint().cut() << std::endl;
  std::cout << " Discriminant is " << perf.workingPoint().discriminantName() << std::endl;
  std::cout << " Is cut based WP " << perf.workingPoint().cutBased() << std::endl;

  p.insert(BinningVariables::JetEta, 0.6);
  p.insert(BinningVariables::Discriminator, 0.23);

  std::cout << " test eta=0.6, discrim = 0.3" << std::endl;
  std::cout << " beff/berr ?" << perf.isResultOk(PerformanceResult::BTAGBEFF, p) << "/"
            << perf.isResultOk(PerformanceResult::BTAGBERR, p) << std::endl;
  std::cout << " beff/berr =" << perf.getResult(PerformanceResult::BTAGBEFF, p) << "/"
            << perf.getResult(PerformanceResult::BTAGBERR, p) << std::endl;
  std::cout << " bSF/bFSerr ?" << perf.isResultOk(PerformanceResult::BTAGBEFFCORR, p) << "/"
            << perf.isResultOk(PerformanceResult::BTAGBERRCORR, p) << std::endl;
  std::cout << " bSF/bSFerr =" << perf.getResult(PerformanceResult::BTAGBEFFCORR, p) << "/"
            << perf.getResult(PerformanceResult::BTAGBERRCORR, p) << std::endl;

  std::cout << std::endl;

  p.insert(BinningVariables::JetEta, 1.8);
  p.insert(BinningVariables::Discriminator, 0.53);

  std::cout << " NEW POINT " << std::endl;

  std::cout << " test eta=1.8, discrim = 0.3" << std::endl;
  std::cout << " beff/berr ?" << perf.isResultOk(PerformanceResult::BTAGBEFF, p) << "/"
            << perf.isResultOk(PerformanceResult::BTAGBERR, p) << std::endl;
  std::cout << " beff/berr =" << perf.getResult(PerformanceResult::BTAGBEFF, p) << "/"
            << perf.getResult(PerformanceResult::BTAGBERR, p) << std::endl;
  std::cout << " bSF/bFSerr ?" << perf.isResultOk(PerformanceResult::BTAGBEFFCORR, p) << "/"
            << perf.isResultOk(PerformanceResult::BTAGBERRCORR, p) << std::endl;
  std::cout << " bSF/bSFerr =" << perf.getResult(PerformanceResult::BTAGBEFFCORR, p) << "/"
            << perf.getResult(PerformanceResult::BTAGBERRCORR, p) << std::endl;

  p.insert(BinningVariables::JetEta, 0.8);
  p.insert(BinningVariables::Discriminator, 1.64);

  std::cout << std::endl;

  std::cout << " NEW POINT " << std::endl;

  std::cout << " test eta=3.8, discrim = 0.4" << std::endl;
  std::cout << " beff/berr ?" << perf.isResultOk(PerformanceResult::BTAGBEFF, p) << "/"
            << perf.isResultOk(PerformanceResult::BTAGBERR, p) << std::endl;
  std::cout << " beff/berr =" << perf.getResult(PerformanceResult::BTAGBEFF, p) << "/"
            << perf.getResult(PerformanceResult::BTAGBERR, p) << std::endl;
  std::cout << " bSF/bFSerr ?" << perf.isResultOk(PerformanceResult::BTAGBEFFCORR, p) << "/"
            << perf.isResultOk(PerformanceResult::BTAGBERRCORR, p) << std::endl;
  std::cout << " bSF/bSFerr =" << perf.getResult(PerformanceResult::BTAGBEFFCORR, p) << "/"
            << perf.getResult(PerformanceResult::BTAGBERRCORR, p) << std::endl;

  /*
  p.insert(BinningVariables::JetEta,0.6);
  p.insert(BinningVariables::Discriminator,2.3);
  
  std::cout <<" FROM MC test eta=0.6, discrim = 0.3"<<std::endl;
  std::cout <<" beff/berr ?"<<perf.isResultOk(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGBERR,p)<<std::endl;
  std::cout <<" beff/berr ="<<perf.getResult(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.getResult(PerformanceResult::BTAGBERR,p)<<std::endl;
  std::cout <<" ceff/cerr ?"<<perf.isResultOk(PerformanceResult::BTAGCEFF,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGCERR,p)<<std::endl;
  std::cout <<" ceff/cerr ="<<perf.getResult(PerformanceResult::BTAGCEFF,p)<<"/"<<perf.getResult(PerformanceResult::BTAGCERR,p)<<std::endl;
  
  std::cout << std::endl;
  */

  //++++++++++++------  TESTING FOR CONTINIOUS DISCRIMINATORS      --------+++++++++++++

  /*
  std::cout <<" The WP is defined by a cut at "<<perf.workingPoint().cut()<<std::endl;
  std::cout <<" Discriminant is "<<perf.workingPoint().discriminantName()<<std::endl;

  std::cout <<" test eta=0.6, et=55"<<std::endl;
  p.insert(BinningVariables::JetEta,0.6);
  p.insert(BinningVariables::JetEt,55);
  std::cout <<" bSF/bSFerr ?"<<perf.isResultOk(PerformanceResult::BTAGLEFFCORR,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGLERRCORR,p)<<std::endl;
  std::cout <<" bSF/bSFerr ="<<perf.getResult(PerformanceResult::BTAGLEFFCORR,p)<<"/"<<perf.getResult(PerformanceResult::BTAGLERRCORR,p)<<std::endl;
  std::cout <<" beff/berr ?"<<perf.isResultOk(PerformanceResult::BTAGLEFF,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGLERR,p)<<std::endl;
  std::cout <<" bSF/bSFerr ="<<perf.getResult(PerformanceResult::BTAGLEFF,p)<<"/"<<perf.getResult(PerformanceResult::BTAGLERR,p)<<std::endl;
  */

  /*
  std::cout <<" test eta=1.9, et=33"<<std::endl;
  p.insert(BinningVariables::JetEta,1.9);
  p.insert(BinningVariables::JetEt,33);
  std::cout <<" beff/berr ?"<<perf.isResultOk(PerformanceResult::BTAGLEFFCORR,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGLERRCORR,p)<<std::endl;
  std::cout <<" beff/berr ="<<perf.getResult(PerformanceResult::BTAGLEFFCORR,p)<<"/"<<perf.getResult(PerformanceResult::BTAGLERRCORR,p)<<std::endl;


  std::cout <<" test eta=0.2, et=433"<<std::endl;
  p.insert(BinningVariables::JetEta,0.9);
  p.insert(BinningVariables::JetEt,433);
  std::cout <<" beff/berr ?"<<perf.isResultOk(PerformanceResult::BTAGLEFFCORR,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGLERRCORR,p)<<std::endl;
  std::cout <<" beff/berr ="<<perf.getResult(PerformanceResult::BTAGLEFFCORR,p)<<"/"<<perf.getResult(PerformanceResult::BTAGLERRCORR,p)<<std::endl;
  */

  //  std::cout <<" now I ask for a calibration which is not present ---> should throw an exception "<<std::endl;

  //  edm::ESHandle<BtagPerformance> perfH2;
  //  iSetup.get<BTagPerformanceRecord>().get("TrackCountingHighEff_tight",perfH2);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestPerformanceFW_ES_TFormula);
