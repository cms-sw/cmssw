/** 
 *  Compare configuration from DB with Python
 *
 *  \author S. Dildick - Texas A&M University
 */

#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Configuration via EventSetup
#include "CondFormats/CSCObjects/interface/CSCDBL1TPParameters.h"
#include "CondFormats/DataRecord/interface/CSCDBL1TPParametersRcd.h"

class L1CSCTPEmulatorConfigAnalyzer : public edm::one::EDAnalyzer<> {
public:
  L1CSCTPEmulatorConfigAnalyzer(const edm::ParameterSet& pset);

  ~L1CSCTPEmulatorConfigAnalyzer() override {}

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  edm::ParameterSet pyConfig;
};

using namespace std;
L1CSCTPEmulatorConfigAnalyzer::L1CSCTPEmulatorConfigAnalyzer(const edm::ParameterSet& iConfig) { pyConfig = iConfig; }

void L1CSCTPEmulatorConfigAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  edm::ESHandle<CSCDBL1TPParameters> confH;
  iSetup.get<CSCDBL1TPParametersRcd>().get(confH);
  const CSCDBL1TPParameters* dbConfig = confH.product();

  // python params
  edm::ParameterSet tmbParams = pyConfig.getParameter<edm::ParameterSet>("tmbParam");
  edm::ParameterSet alctParams = pyConfig.getParameter<edm::ParameterSet>("alctParam07");
  edm::ParameterSet clctParams = pyConfig.getParameter<edm::ParameterSet>("clctParam07");

  unsigned int db_alctFifoTbins = dbConfig->alctFifoTbins();
  unsigned int db_alctFifoPretrig = dbConfig->alctFifoPretrig();
  unsigned int db_alctDriftDelay = dbConfig->alctDriftDelay();
  unsigned int db_alctNplanesHitPretrig = dbConfig->alctNplanesHitPretrig();
  unsigned int db_alctNplanesHitPattern = dbConfig->alctNplanesHitPattern();
  unsigned int db_alctNplanesHitAccelPretrig = dbConfig->alctNplanesHitAccelPretrig();
  unsigned int db_alctNplanesHitAccelPattern = dbConfig->alctNplanesHitAccelPattern();
  unsigned int db_alctTrigMode = dbConfig->alctTrigMode();
  unsigned int db_alctAccelMode = dbConfig->alctAccelMode();
  unsigned int db_alctL1aWindowWidth = dbConfig->alctL1aWindowWidth();

  unsigned int db_clctFifoTbins = dbConfig->clctFifoTbins();
  unsigned int db_clctFifoPretrig = dbConfig->clctFifoPretrig();
  unsigned int db_clctHitPersist = dbConfig->clctHitPersist();
  unsigned int db_clctDriftDelay = dbConfig->clctDriftDelay();
  unsigned int db_clctNplanesHitPretrig = dbConfig->clctNplanesHitPretrig();
  unsigned int db_clctNplanesHitPattern = dbConfig->clctNplanesHitPattern();
  unsigned int db_clctPidThreshPretrig = dbConfig->clctPidThreshPretrig();
  unsigned int db_clctMinSeparation = dbConfig->clctMinSeparation();

  unsigned int db_tmbMpcBlockMe1a = dbConfig->tmbMpcBlockMe1a();
  unsigned int db_tmbAlctTrigEnable = dbConfig->tmbAlctTrigEnable();
  unsigned int db_tmbClctTrigEnable = dbConfig->tmbClctTrigEnable();
  unsigned int db_tmbMatchTrigEnable = dbConfig->tmbMatchTrigEnable();
  unsigned int db_tmbMatchTrigWindowSize = dbConfig->tmbMatchTrigWindowSize();
  unsigned int db_tmbL1aWindowSize = dbConfig->tmbTmbL1aWindowSize();

  unsigned int py_alctFifoTbins = alctParams.getParameter<unsigned int>("alctFifoTbins");
  unsigned int py_alctFifoPretrig = alctParams.getParameter<unsigned int>("alctFifoPretrig");
  unsigned int py_alctDriftDelay = alctParams.getParameter<unsigned int>("alctDriftDelay");
  unsigned int py_alctNplanesHitPretrig = alctParams.getParameter<unsigned int>("alctNplanesHitPretrig");
  unsigned int py_alctNplanesHitPattern = alctParams.getParameter<unsigned int>("alctNplanesHitPattern");
  unsigned int py_alctNplanesHitAccelPretrig = alctParams.getParameter<unsigned int>("alctNplanesHitAccelPretrig");
  unsigned int py_alctNplanesHitAccelPattern = alctParams.getParameter<unsigned int>("alctNplanesHitAccelPattern");
  unsigned int py_alctTrigMode = alctParams.getParameter<unsigned int>("alctTrigMode");
  unsigned int py_alctAccelMode = alctParams.getParameter<unsigned int>("alctAccelMode");
  unsigned int py_alctL1aWindowWidth = alctParams.getParameter<unsigned int>("alctL1aWindowWidth");

  unsigned int py_clctFifoTbins = clctParams.getParameter<unsigned int>("clctFifoTbins");
  unsigned int py_clctFifoPretrig = clctParams.getParameter<unsigned int>("clctFifoPretrig");
  unsigned int py_clctHitPersist = clctParams.getParameter<unsigned int>("clctHitPersist");
  unsigned int py_clctDriftDelay = clctParams.getParameter<unsigned int>("clctDriftDelay");
  unsigned int py_clctNplanesHitPretrig = clctParams.getParameter<unsigned int>("clctNplanesHitPretrig");
  unsigned int py_clctNplanesHitPattern = clctParams.getParameter<unsigned int>("clctNplanesHitPattern");
  unsigned int py_clctPidThreshPretrig = clctParams.getParameter<unsigned int>("clctPidThreshPretrig");
  unsigned int py_clctMinSeparation = clctParams.getParameter<unsigned int>("clctMinSeparation");

  unsigned int py_tmbMpcBlockMe1a = tmbParams.getParameter<unsigned int>("mpcBlockMe1a");
  unsigned int py_tmbAlctTrigEnable = tmbParams.getParameter<unsigned int>("alctTrigEnable");
  unsigned int py_tmbClctTrigEnable = tmbParams.getParameter<unsigned int>("clctTrigEnable");
  unsigned int py_tmbMatchTrigEnable = tmbParams.getParameter<unsigned int>("matchTrigEnable");
  unsigned int py_tmbMatchTrigWindowSize = tmbParams.getParameter<unsigned int>("matchTrigWindowSize");
  unsigned int py_tmbL1aWindowSize = tmbParams.getParameter<unsigned int>("tmbL1aWindowSize");

  //check
  bool ok_alctFifoTbins = db_alctFifoTbins == py_alctFifoTbins;
  bool ok_alctFifoPretrig = db_alctFifoPretrig == py_alctFifoPretrig;
  bool ok_alctDriftDelay = db_alctDriftDelay == py_alctDriftDelay;
  bool ok_alctNplanesHitPretrig = db_alctNplanesHitPretrig == py_alctNplanesHitPretrig;
  bool ok_alctNplanesHitPattern = db_alctNplanesHitPattern == py_alctNplanesHitPattern;
  bool ok_alctNplanesHitAccelPretrig = db_alctNplanesHitAccelPretrig == py_alctNplanesHitAccelPretrig;
  bool ok_alctNplanesHitAccelPattern = db_alctNplanesHitAccelPattern == py_alctNplanesHitAccelPattern;
  bool ok_alctTrigMode = db_alctTrigMode == py_alctTrigMode;
  bool ok_alctAccelMode = db_alctAccelMode == py_alctAccelMode;
  bool ok_alctL1aWindowWidth = db_alctL1aWindowWidth == py_alctL1aWindowWidth;

  bool ok_clctFifoTbins = db_clctFifoTbins == py_clctFifoTbins;
  bool ok_clctFifoPretrig = db_clctFifoPretrig == py_clctFifoPretrig;
  bool ok_clctHitPersist = db_clctHitPersist == py_clctHitPersist;
  bool ok_clctDriftDelay = db_clctDriftDelay == py_clctDriftDelay;
  bool ok_clctNplanesHitPretrig = db_clctNplanesHitPretrig == py_clctNplanesHitPretrig;
  bool ok_clctNplanesHitPattern = db_clctNplanesHitPattern == py_clctNplanesHitPattern;
  bool ok_clctPidThreshPretrig = db_clctPidThreshPretrig == py_clctPidThreshPretrig;
  bool ok_clctMinSeparation = db_clctMinSeparation == py_clctMinSeparation;

  bool ok_tmbMpcBlockMe1a = db_tmbMpcBlockMe1a == py_tmbMpcBlockMe1a;
  bool ok_tmbAlctTrigEnable = db_tmbAlctTrigEnable == py_tmbAlctTrigEnable;
  bool ok_tmbClctTrigEnable = db_tmbClctTrigEnable == py_tmbClctTrigEnable;
  bool ok_tmbMatchTrigEnable = db_tmbMatchTrigEnable == py_tmbMatchTrigEnable;
  bool ok_tmbMatchTrigWindowSize = db_tmbMatchTrigWindowSize == py_tmbMatchTrigWindowSize;
  bool ok_tmbL1aWindowSize = db_tmbL1aWindowSize == py_tmbL1aWindowSize;

  std::cout << std::endl;
  std::cout << "Start Comparing the L1 CSC TP emulator settings between Python and conditions DB." << std::endl;

  std::cout << std::endl;
  std::cout << "Parameters different between Py and DB" << std::endl;
  std::cout << "- - - - - - - - - - - - - - - - - - - " << std::endl;
  std::cout << std::endl;

  if (!ok_alctFifoTbins)
    std::cout << "alctFifoTbins: Py = " << py_alctFifoTbins << ", DB = " << db_alctFifoTbins << std::endl;
  if (!ok_alctFifoPretrig)
    std::cout << "alctFifoPretrig: Py = " << py_alctFifoPretrig << ", DB = " << db_alctFifoPretrig << std::endl;
  if (!ok_alctDriftDelay)
    std::cout << "alctDriftDelay: Py = " << py_alctDriftDelay << ", DB = " << db_alctDriftDelay << std::endl;
  if (!ok_alctNplanesHitPretrig)
    std::cout << "alctNplanesHitPretrig: Py = " << py_alctNplanesHitPretrig << ", DB = " << db_alctNplanesHitPretrig
              << std::endl;
  if (!ok_alctNplanesHitPattern)
    std::cout << "alctNplanesHitPattern: Py = " << py_alctNplanesHitPattern << ", DB = " << db_alctNplanesHitPattern
              << std::endl;
  if (!ok_alctNplanesHitAccelPretrig)
    std::cout << "alctNplanesHitAccelPretrig: Py = " << py_alctNplanesHitAccelPretrig
              << ", DB = " << db_alctNplanesHitAccelPretrig << std::endl;
  if (!ok_alctNplanesHitAccelPattern)
    std::cout << "alctNplanesHitAccelPattern: Py = " << py_alctNplanesHitAccelPattern
              << ", DB = " << db_alctNplanesHitAccelPattern << std::endl;
  if (!ok_alctTrigMode)
    std::cout << "alctTrigMode: Py = " << py_alctTrigMode << ", DB = " << db_alctTrigMode << std::endl;
  if (!ok_alctAccelMode)
    std::cout << "alctAccelMode: Py = " << py_alctAccelMode << ", DB = " << db_alctAccelMode << std::endl;
  if (!ok_alctL1aWindowWidth)
    std::cout << "alctL1aWindowWidth: Py = " << py_alctL1aWindowWidth << ", DB = " << db_alctL1aWindowWidth
              << std::endl;
  std::cout << std::endl;

  if (!ok_clctFifoTbins)
    std::cout << "clctFifoTbins: Py = " << py_clctFifoTbins << ", DB = " << db_clctFifoTbins << std::endl;
  if (!ok_clctFifoPretrig)
    std::cout << "clctFifoPretrig: Py = " << py_clctFifoPretrig << ", DB = " << db_clctFifoPretrig << std::endl;
  if (!ok_clctHitPersist)
    std::cout << "clctHitPersist: Py = " << py_clctHitPersist << ", DB = " << db_clctHitPersist << std::endl;
  if (!ok_clctDriftDelay)
    std::cout << "clctDriftDelay: Py = " << py_clctDriftDelay << ", DB = " << db_clctDriftDelay << std::endl;
  if (!ok_clctNplanesHitPretrig)
    std::cout << "clctNplanesHitPretrig: Py = " << py_clctNplanesHitPretrig << ", DB = " << db_clctNplanesHitPretrig
              << std::endl;
  if (!ok_clctNplanesHitPattern)
    std::cout << "clctNplanesHitPattern: Py = " << py_clctNplanesHitPattern << ", DB = " << db_clctNplanesHitPattern
              << std::endl;
  if (!ok_clctPidThreshPretrig)
    std::cout << "clctPidThreshPretrig: Py = " << py_clctPidThreshPretrig << ", DB = " << db_clctPidThreshPretrig
              << std::endl;
  if (!ok_clctMinSeparation)
    std::cout << "clctMinSeparation: Py = " << py_clctMinSeparation << ", DB = " << db_clctMinSeparation << std::endl;
  std::cout << std::endl;

  if (!ok_tmbMpcBlockMe1a)
    std::cout << "tmbMpcBlockMe1a: Py = " << py_tmbMpcBlockMe1a << ", DB = " << db_tmbMpcBlockMe1a << std::endl;
  if (!ok_tmbAlctTrigEnable)
    std::cout << "tmbAlctTrigEnable: Py = " << py_tmbAlctTrigEnable << ", DB = " << db_tmbAlctTrigEnable << std::endl;
  if (!ok_tmbClctTrigEnable)
    std::cout << "tmbClctTrigEnable: Py = " << py_tmbClctTrigEnable << ", DB = " << db_tmbClctTrigEnable << std::endl;
  if (!ok_tmbMatchTrigEnable)
    std::cout << "tmbMatchTrigEnable: Py = " << py_tmbMatchTrigEnable << ", DB = " << db_tmbMatchTrigEnable
              << std::endl;
  if (!ok_tmbMatchTrigWindowSize)
    std::cout << "tmbMatchTrigWindowSize: Py = " << py_tmbMatchTrigWindowSize << ", DB = " << db_tmbMatchTrigWindowSize
              << std::endl;
  if (!ok_tmbL1aWindowSize)
    std::cout << "tmbL1aWindowSize: Py = " << py_tmbL1aWindowSize << ", DB = " << db_tmbL1aWindowSize << std::endl;

  std::cout << std::endl;
  std::cout << "Parameters same in Py and DB" << std::endl;
  std::cout << "- - - - - - - - - - - - - - " << std::endl;
  std::cout << std::endl;

  if (ok_alctFifoTbins)
    std::cout << "alctFifoTbins: " << py_alctFifoTbins << std::endl;
  if (ok_alctFifoPretrig)
    std::cout << "alctFifoPretrig: " << py_alctFifoPretrig << std::endl;
  if (ok_alctDriftDelay)
    std::cout << "alctDriftDelay: " << py_alctDriftDelay << std::endl;
  if (ok_alctNplanesHitPretrig)
    std::cout << "alctNplanesHitPretrig: " << py_alctNplanesHitPretrig << std::endl;
  if (ok_alctNplanesHitPattern)
    std::cout << "alctNplanesHitPattern: " << py_alctNplanesHitPattern << std::endl;
  if (ok_alctNplanesHitAccelPretrig)
    std::cout << "alctNplanesHitAccelPretrig: " << py_alctNplanesHitAccelPretrig << std::endl;
  if (ok_alctNplanesHitAccelPattern)
    std::cout << "alctNplanesHitAccelPattern: " << py_alctNplanesHitAccelPattern << std::endl;
  if (ok_alctTrigMode)
    std::cout << "alctTrigMode: " << py_alctTrigMode << std::endl;
  if (ok_alctAccelMode)
    std::cout << "alctAccelMode: " << py_alctAccelMode << std::endl;
  if (ok_alctL1aWindowWidth)
    std::cout << "alctL1aWindowWidth: " << py_alctL1aWindowWidth << std::endl;
  std::cout << std::endl;

  if (ok_clctFifoTbins)
    std::cout << "clctFifoTbins: " << py_clctFifoTbins << std::endl;
  if (ok_clctFifoPretrig)
    std::cout << "clctFifoPretrig: " << py_clctFifoPretrig << std::endl;
  if (ok_clctHitPersist)
    std::cout << "clctHitPersist: " << py_clctHitPersist << std::endl;
  if (ok_clctDriftDelay)
    std::cout << "clctDriftDelay: " << py_clctDriftDelay << std::endl;
  if (ok_clctNplanesHitPretrig)
    std::cout << "clctNplanesHitPretrig: " << py_clctNplanesHitPretrig << std::endl;
  if (ok_clctNplanesHitPattern)
    std::cout << "clctNplanesHitPattern: " << py_clctNplanesHitPattern << std::endl;
  if (ok_clctPidThreshPretrig)
    std::cout << "clctPidThreshPretrig: " << py_clctPidThreshPretrig << std::endl;
  if (ok_clctMinSeparation)
    std::cout << "clctMinSeparation: " << py_clctMinSeparation << std::endl;
  std::cout << std::endl;

  if (ok_tmbMpcBlockMe1a)
    std::cout << "tmbMpcBlockMe1a: " << py_tmbMpcBlockMe1a << std::endl;
  if (ok_tmbAlctTrigEnable)
    std::cout << "tmbAlctTrigEnable: " << py_tmbAlctTrigEnable << std::endl;
  if (ok_tmbClctTrigEnable)
    std::cout << "tmbClctTrigEnable: " << py_tmbClctTrigEnable << std::endl;
  if (ok_tmbMatchTrigEnable)
    std::cout << "tmbMatchTrigEnable: " << py_tmbMatchTrigEnable << std::endl;
  if (ok_tmbMatchTrigWindowSize)
    std::cout << "tmbMatchTrigWindowSize: " << py_tmbMatchTrigWindowSize << std::endl;
  if (ok_tmbL1aWindowSize)
    std::cout << "tmbL1aWindowSize: " << py_tmbL1aWindowSize << std::endl;
  std::cout << std::endl;

  std::cout << "Done." << std::endl;
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1CSCTPEmulatorConfigAnalyzer);
