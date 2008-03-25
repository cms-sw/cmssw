#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include <CalibCalorimetry/HcalStandardModules/interface/HcalLedAnalyzer.h>
#include "CalibCalorimetry/HcalAlgos/interface/HcalAlgoUtils.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
//#include "CondTools/Hcal/interface/HcalDbTool.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"

#include <iostream>
#include <fstream>

namespace {
  bool defaultsFile (const std::string fParam) {
    return fParam == "defaults";
  }

  bool asciiFile (const std::string fParam) {
    return fParam.find (':') == std::string::npos && std::string (fParam, fParam.length () - 4) == ".txt";
  }
  
  bool xmlFile (const std::string fParam) {
    return fParam.find (':') == std::string::npos && std::string (fParam, fParam.length () - 4) == ".xml";
  }
  
  bool dbFile (const std::string fParam) {
    return fParam.find (':') != std::string::npos;
  }

  template <class T> 
  bool getObject (T* fObject, const std::string& fDb, const std::string& fTag, int fRun) {
    if (!fObject) return false;
    if (fDb.empty ()) return false; 
    if (asciiFile (fDb)) {
      std::cout << "HcalLedAnalyzer-> USE INPUT: ASCII " << std::endl;
      std::ifstream stream (fDb.c_str ());
      HcalDbASCIIIO::getObject (stream, fObject); 
      return true;
    }
    else if (dbFile (fDb)) {
      std::cout << "HcalLedAnalyzer-> USE INPUT: Pool " << fDb << std::endl;
      std::cout << "HcalPedestalAnalyzer-> Pool interface is not supportet since 1.3.0" << fDb << std::endl;
      return false;
//       HcalDbTool poolDb (fDb);
//       return poolDb.getObject (fObject, fTag, fRun);
    }
    else {
      std::cerr << "HcalLedAnalyzer-> Unknown input type " << fDb << std::endl;
      return false;
    }
  }
  
}

HcalLedAnalyzer::HcalLedAnalyzer(const edm::ParameterSet& ps){
  m_ledAnal = new HcalLedAnalysis(ps);
  m_ledAnal->LedSetup(ps.getUntrackedParameter<std::string>("outputFileHist", "HcalLedAnalyzer.root"));
//  m_startSample = ps.getUntrackedParameter<int>("firstSample", 0);
//  m_endSample = ps.getUntrackedParameter<int>("lastSample", 19);
  m_inputPedestals_source = ps.getUntrackedParameter<std::string>("inputPedestalsSource", "");
  m_inputPedestals_tag = ps.getUntrackedParameter<std::string>("inputPedsTag", "");
  m_inputPedestals_run = ps.getUntrackedParameter<int>("inputPedsRun", 1);

  // CORAL required variables to be set, even if not needed
  if (!::getenv("CORAL_AUTH_USER")) ::putenv("CORAL_AUTH_USER=blah");
  if (!::getenv("CORAL_AUTH_PASSWORD")) ::putenv("CORAL_AUTH_PASSWORD=blah"); 
}

HcalLedAnalyzer::~HcalLedAnalyzer(){
//  delete m_ledAnal;
}

void HcalLedAnalyzer::beginJob(const edm::EventSetup& c){
  m_ievt = 0;
  led_sample = 1;
  HcalPedestals* inputPeds=0;
// get pedestals
  if (!m_inputPedestals_source.empty ()) {
    inputPeds = new HcalPedestals ();
    if (!getObject (inputPeds, m_inputPedestals_source, m_inputPedestals_tag, m_inputPedestals_run)) {
      std::cerr << "HcalLedAnalyzer-> Failed to get pedestal values" << std::endl;
    }
    m_ledAnal->doPeds(inputPeds);
    delete inputPeds;
  }
}

void HcalLedAnalyzer::endJob(void) {
  m_ledAnal->LedDone();
  std::cout<<"Getting out"<<std::endl;
}

void HcalLedAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& eventSetup){

  m_ievt++;

  ///get digis
  edm::Handle<HBHEDigiCollection> hbhe; e.getByType(hbhe);
  edm::Handle<HODigiCollection> ho;     e.getByType(ho);
  edm::Handle<HFDigiCollection> hf;     e.getByType(hf);

  // get conditions
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);

  m_ledAnal->processLedEvent(*hbhe, *ho, *hf, *conditions);

  if(m_ievt%1000 == 0)
    std::cout << "HcalLedAnalyzer: analyzed " << m_ievt << " events" << std::endl;

  return;
}

