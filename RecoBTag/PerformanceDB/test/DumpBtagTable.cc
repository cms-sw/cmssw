// -*- C++ -*-
//
// Package:    DumpBtagTable
// Class:      DumpBtagTable
//
/**\class DumpBtagTable DumpBtagTable.cc RecoBTag/DumpBtagTable/src/DumpBtagTable.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
// Modified: Francisco Yumiceva
// Original Author:  Tommaso Boccali
//         Created:  Tue Nov 25 15:50:50 CET 2008
//
//

// system include files
#include <iostream>
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

using namespace std;

class DumpBtagTable : public edm::EDAnalyzer {
public:
  explicit DumpBtagTable(const edm::ParameterSet&);
  ~DumpBtagTable() override;

private:
  string name;
  vector<string> measureName;
  vector<string> measureType;
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
};

DumpBtagTable::DumpBtagTable(const edm::ParameterSet& iConfig)

{
  measureName = iConfig.getParameter<vector<string> >("measureName");
  measureType = iConfig.getParameter<vector<string> >("measureType");
}

DumpBtagTable::~DumpBtagTable() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void DumpBtagTable::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  map<string, PerformanceResult::ResultType> measureMap;
  measureMap["BTAGBEFF"] = PerformanceResult::BTAGBEFF;
  measureMap["BTAGBERR"] = PerformanceResult::BTAGBERR;
  measureMap["BTAGCEFF"] = PerformanceResult::BTAGCEFF;
  measureMap["BTAGCERR"] = PerformanceResult::BTAGCERR;
  measureMap["BTAGLEFF"] = PerformanceResult::BTAGLEFF;
  measureMap["BTAGLERR"] = PerformanceResult::BTAGLERR;
  measureMap["BTAGNBEFF"] = PerformanceResult::BTAGNBEFF;
  measureMap["BTAGNBERR"] = PerformanceResult::BTAGNBERR;
  measureMap["BTAGBEFFCORR"] = PerformanceResult::BTAGBEFFCORR;
  measureMap["BTAGBERRCORR"] = PerformanceResult::BTAGBERRCORR;
  measureMap["BTAGCEFFCORR"] = PerformanceResult::BTAGCEFFCORR;
  measureMap["BTAGCERRCORR"] = PerformanceResult::BTAGCERRCORR;
  measureMap["BTAGLEFFCORR"] = PerformanceResult::BTAGLEFFCORR;
  measureMap["BTAGLERRCORR"] = PerformanceResult::BTAGLERRCORR;
  measureMap["BTAGNBEFFCORR"] = PerformanceResult::BTAGNBEFFCORR;
  measureMap["BTAGNBERRCORR"] = PerformanceResult::BTAGNBERRCORR;
  measureMap["BTAGNBERRCORR"] = PerformanceResult::BTAGNBERRCORR;
  measureMap["MUEFF"] = PerformanceResult::MUEFF;
  measureMap["MUERR"] = PerformanceResult::MUERR;
  measureMap["MUFAKE"] = PerformanceResult::MUFAKE;
  measureMap["MUEFAKE"] = PerformanceResult::MUEFAKE;

  edm::ESHandle<BtagPerformance> perfH;
  if (measureName.size() != measureType.size()) {
    std::cout << "measureName, measureType size mismatch!" << std::endl;
    exit(-1);
  }

  for (size_t iMeasure = 0; iMeasure < measureName.size(); iMeasure++) {
    std::cout << "# Dump table: " << measureName[iMeasure] << " of type " << measureType[iMeasure] << std::endl;

    //Setup our measurement
    iSetup.get<BTagPerformanceRecord>().get(measureName[iMeasure], perfH);
    const BtagPerformance& perf = *(perfH.product());

    //Working point
    std::cout << "# Working point: " << perf.workingPoint().cut() << std::endl;
    std::cout << "# [pt] [eta] [SF]" << std::endl;

    //Setup the point we wish to test!

    BinningPointByMap measurePoint;

    string sep = " ";

    float bin_pt = 10.;
    float bin_eta = 1.2;

    for (int i = 2; i < 50; i++) {
      for (int j = 0; j < 2; j++) {
        float pt = float(i * bin_pt);
        float eta = float(j * bin_eta);

        measurePoint.insert(BinningVariables::JetEt, pt);
        measurePoint.insert(BinningVariables::JetAbsEta, eta);

        //std::cout << " From performance DB: " << perf.isResultOk( measureMap[ measureType[ iMeasure] ], measurePoint)
        //<< " result at Jet pT " << pt << " GeV, |eta| < 2.39 " << perf.getResult( measureMap[ measureType[ iMeasure] ], measurePoint)
        //<< std::endl;
        std::cout << pt << sep << pt + bin_pt << sep << eta << sep << eta + bin_eta << sep
                  << perf.getResult(measureMap[measureType[iMeasure]], measurePoint) << std::endl;

        measurePoint.reset();
      }
    }

    measurePoint.reset();
    //measurePoint.insert(BinningVariables::JetEt,30);
    //measurePoint.insert(BinningVariables::JetAbsEta,2.39);

    //std::cout << "Is it OK? " << perf.isResultOk( measureMap[ measureType[ iMeasure] ], measurePoint)
    //		<< " result at 30 GeV, |eta| < 2.39 " << perf.getResult( measureMap[ measureType[ iMeasure] ], measurePoint)
    //	<< std::endl;

    /*
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
      */
  }
}

// ------------ method called once each job just before starting event loop  ------------
void DumpBtagTable::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void DumpBtagTable::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(DumpBtagTable);
