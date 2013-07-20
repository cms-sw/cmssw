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
// $Id: TestPerformanceFW_ES.cc,v 1.7 2013/01/31 17:54:44 msegala Exp $
//
//


// system include files
#include <memory>
#include <map>
#include <vector>
#include <string>

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

class TestPerformanceFW_ES : public edm::EDAnalyzer {
public:
  explicit TestPerformanceFW_ES(const edm::ParameterSet&);
  ~TestPerformanceFW_ES();
  
  
private:
  std::string name;
  std::vector<std::string> measureName;
  std::vector<std::string> measureType;
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------
};


TestPerformanceFW_ES::TestPerformanceFW_ES(const edm::ParameterSet& iConfig)

{
  measureName = iConfig.getParameter< std::vector< std::string > >("measureName");
  measureType = iConfig.getParameter< std::vector< std::string > >("measureType");
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
  std::map<std::string,PerformanceResult::ResultType> measureMap;
  measureMap["BTAGBEFF"]=PerformanceResult::BTAGBEFF;
  measureMap["BTAGBERR"]=PerformanceResult::BTAGBERR;
  measureMap["BTAGCEFF"]=PerformanceResult::BTAGCEFF;
  measureMap["BTAGCERR"]=PerformanceResult::BTAGCERR;
  measureMap["BTAGLEFF"]=PerformanceResult::BTAGLEFF;
  measureMap["BTAGLERR"]=PerformanceResult::BTAGLERR;
  measureMap["BTAGNBEFF"]=PerformanceResult::BTAGNBEFF;
  measureMap["BTAGNBERR"]=PerformanceResult::BTAGNBERR;
  measureMap["BTAGBEFFCORR"]=PerformanceResult::BTAGBEFFCORR;
  measureMap["BTAGBERRCORR"]=PerformanceResult::BTAGBERRCORR;
  measureMap["BTAGCEFFCORR"]=PerformanceResult::BTAGCEFFCORR;
  measureMap["BTAGCERRCORR"]=PerformanceResult::BTAGCERRCORR;
  measureMap["BTAGLEFFCORR"]=PerformanceResult::BTAGLEFFCORR;
  measureMap["BTAGLERRCORR"]=PerformanceResult::BTAGLERRCORR;
  measureMap["BTAGNBEFFCORR"]=PerformanceResult::BTAGNBEFFCORR;
  measureMap["BTAGNBERRCORR"]=PerformanceResult::BTAGNBERRCORR;
  measureMap["BTAGNBERRCORR"]=PerformanceResult::BTAGNBERRCORR;
  measureMap["MUEFF"]=PerformanceResult::MUEFF;
  measureMap["MUERR"]=PerformanceResult::MUERR;
  measureMap["MUFAKE"]=PerformanceResult::MUFAKE; 
  measureMap["MUEFAKE"]=PerformanceResult::MUEFAKE;


  edm::ESHandle<BtagPerformance> perfH;
  if( measureName.size() != measureType.size() )
  {
      std::cout << "measureName, measureType size mismatch!" << std::endl;
      exit(-1);
  }


  for( size_t iMeasure = 0; iMeasure < measureName.size(); iMeasure++ )
  {
      std::cout << "Testing: " << measureName[ iMeasure ] << " of type " << measureType[ iMeasure ] << std::endl;

//Setup our measurement
      iSetup.get<BTagPerformanceRecord>().get( measureName[ iMeasure ],perfH);
      const BtagPerformance & perf = *(perfH.product());

//Working point
      std::cout << "Working point: " << perf.workingPoint().cut() << std::endl;
//Setup the point we wish to test!
      BinningPointByMap measurePoint;
      measurePoint.insert(BinningVariables::JetEt,50);
      measurePoint.insert(BinningVariables::JetAbsEta,0.6);

      std::cout << "Is it OK? " << perf.isResultOk( measureMap[ measureType[ iMeasure] ], measurePoint)
		<< " result at 50 GeV, 0,6 |eta| " << perf.getResult( measureMap[ measureType[ iMeasure] ], measurePoint)
		<< std::endl;

      std::cout << "Error checking!" << std::endl;
      measurePoint.reset();
      measurePoint.insert(BinningVariables::JetEt,0);
      measurePoint.insert(BinningVariables::JetAbsEta,10);

      std::cout << "Is it OK? " << perf.isResultOk( measureMap[ measureType[ iMeasure] ], measurePoint)
		<< " result at 0 GeV, 10 |eta| " << perf.getResult( measureMap[ measureType[ iMeasure] ], measurePoint)
		<< std::endl;
      std::cout << std::endl;
  }



  // std::cout << "Values: "<<
  //   PerformanceResult::BTAGNBEFF<<" " <<
  //   PerformanceResult::MUERR<<" " <<
  //   std::endl;

  // // check beff, berr for eta=.6, et=55;
  // BinningPointByMap p;

  // std::cout <<" My Performance Object is indeed a "<<typeid(perfH.product()).name()<<std::endl;

  // std::cout <<" test eta=0.6, et=55"<<std::endl;


  // p.insert(BinningVariables::JetEta,0.6);
  // p.insert(BinningVariables::JetEt,55);
  // std::cout <<" nbeff/nberr ?"<<perf.isResultOk(PerformanceResult::BTAGNBEFF,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGNBERR,p)<<std::endl;
  // std::cout <<" beff/berr ?"<<perf.isResultOk(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGBERR,p)<<std::endl;
  // std::cout <<" beff/berr ="<<perf.getResult(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.getResult(PerformanceResult::BTAGBERR,p)<<std::endl;

  // std::cout <<" test eta=1.9, et=33"<<std::endl;
  //  p.insert(BinningVariables::JetEta,1.9);
  // p.insert(BinningVariables::JetEt,33);
  // std::cout <<" beff/berr ?"<<perf.isResultOk(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGBERR,p)<<std::endl;
  // std::cout <<" beff/berr ="<<perf.getResult(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.getResult(PerformanceResult::BTAGBERR,p)<<std::endl;

  // std::cout <<" The WP is defined by a cut at "<<perf.workingPoint().cut()<<std::endl;
  // std::cout <<" Discriminant is "<<perf.workingPoint().discriminantName()<<std::endl;

  // std::cout <<" now I ask for a calibration but I do not set eta in the binning point ---> should return all not available "<<std::endl;
  // p.reset();
  // p.insert(BinningVariables::JetNTracks,3);
  // p.insert(BinningVariables::JetEt,55);
  // std::cout <<" beff/berr ?"<<perf.isResultOk(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGBERR,p)<<std::endl;
  // std::cout <<" beff/berr ="<<perf.getResult(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.getResult(PerformanceResult::BTAGBERR,p)<<std::endl;

  // //  std::cout <<" now I ask for a calibration which is not present ---> should throw an exception "<<std::endl;

  // //  edm::ESHandle<BtagPerformance> perfH2;
  // //  iSetup.get<BTagPerformanceRecord>().get("TrackCountingHighEff_tight",perfH2);
  

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
