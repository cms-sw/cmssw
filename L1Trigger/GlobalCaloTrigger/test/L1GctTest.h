// -*- C++ -*-
//
// Package:    L1GctTest
// Class:      L1GctTest
// 
/**\class L1GctTest L1GctTest.cc L1Trigger/GlobalCaloTrigger/test/L1GctTest.cc

 Description: a collection of GCT tests

*/
//
// Original Author:  Gregory Heath
//         Created:  Mon Mar 12 16:36:35 CET 2007
// $Id: L1GctTest.h,v 1.4 2008/04/15 10:35:42 heath Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class L1GlobalCaloTrigger;
class L1GctJetEtCalibrationLut;
class gctTestFunctions;

//
// class declaration
//

class L1GctTest : public edm::EDAnalyzer {
public:
  explicit L1GctTest(const edm::ParameterSet&);
  ~L1GctTest();


private:
  virtual void beginJob(const edm::EventSetup& c) ;
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  virtual void endJob() ;

  void configureGct(const edm::EventSetup& c);

  // ----------member data ---------------------------

  L1GlobalCaloTrigger* m_gct;
  L1GctJetEtCalibrationLut* m_jetEtCalibLut;

  gctTestFunctions* m_tester;

  bool theElectronTestIsEnabled;
  bool theEnergyAlgosTestIsEnabled;
  bool theFirmwareTestIsEnabled;

  std::string theInputDataFileName;
  std::string theReferenceDataFileName;
  std::string theEnergySumsDataFileName;

  unsigned m_eventNo;
  bool m_allGood;
};
