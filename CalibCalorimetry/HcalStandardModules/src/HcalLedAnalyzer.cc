#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include <CalibCalorimetry/HcalStandardModules/interface/HcalLedAnalyzer.h>
//#include "CondTools/Hcal/interface/HcalDbTool.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"



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

HcalLedAnalyzer::HcalLedAnalyzer(const edm::ParameterSet& ps) :
  hbheDigiCollectionTag_(ps.getParameter<edm::InputTag>("hbheDigiCollectionTag")),
  hoDigiCollectionTag_(ps.getParameter<edm::InputTag>("hoDigiCollectionTag")),
  hfDigiCollectionTag_(ps.getParameter<edm::InputTag>("hfDigiCollectionTag")),
  hcalCalibDigiCollectionTag_ (ps.getParameter<edm::InputTag>("hcalCalibDigiCollectionTag")) {

  m_ledAnal = new HcalLedAnalysis(ps);
  m_ledAnal->LedSetup(ps.getUntrackedParameter<std::string>("outputFileHist", "HcalLedAnalyzer.root"));
//  m_startSample = ps.getUntrackedParameter<int>("firstSample", 0);
//  m_endSample = ps.getUntrackedParameter<int>("lastSample", 19);
  m_inputPedestals_source = ps.getUntrackedParameter<std::string>("inputPedestalsSource", "");
  m_inputPedestals_tag = ps.getUntrackedParameter<std::string>("inputPedsTag", "");
  m_inputPedestals_run = ps.getUntrackedParameter<int>("inputPedsRun", 1);

  // CORAL required variables to be set, even if not needed
  const char* foo1 = "CORAL_AUTH_USER=blah";
  const char* foo2 = "CORAL_AUTH_PASSWORD=blah";
  if (!::getenv("CORAL_AUTH_USER")) ::putenv(const_cast<char*>(foo1));
  if (!::getenv("CORAL_AUTH_PASSWORD")) ::putenv(const_cast<char*>(foo2)); 
}

HcalLedAnalyzer::~HcalLedAnalyzer(){
//  delete m_ledAnal;
}

void HcalLedAnalyzer::beginJob(){
  m_ievt = 0;
  led_sample = 1;
}

void HcalLedAnalyzer::endJob(void) {
  m_ledAnal->LedDone();
  std::cout<<"Getting out"<<std::endl;
}

void HcalLedAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& eventSetup){

  m_ievt++;

  ///get digis
  edm::Handle<HBHEDigiCollection> hbhe; e.getByLabel(hbheDigiCollectionTag_, hbhe);
  edm::Handle<HODigiCollection> ho;     e.getByLabel(hoDigiCollectionTag_, ho);
  edm::Handle<HFDigiCollection> hf;     e.getByLabel(hfDigiCollectionTag_, hf);

  // get calib digis
  edm::Handle<HcalCalibDigiCollection> calib;  e.getByLabel(hcalCalibDigiCollectionTag_, calib);

  // get testbeam specific laser info from the TDC.  This probably will not work
  // outside of the testbeam, but it should be easy to modify the Handle/getByType
  // to get the correct stuff 

  //edm::Handle<HcalTBTiming> timing; e.getByType(timing);

  // get conditions
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);

//  int runNumber = e.id().run();
//  m_ledAnal->processLedEvent(*hbhe, *ho, *hf, *calib, *conditions, *runNumber);

  m_ledAnal->processLedEvent(*hbhe, *ho, *hf, *calib, *conditions);


  if(m_ievt%1000 == 0)
    std::cout << "HcalLedAnalyzer: analyzed " << m_ievt << " events" << std::endl;

  return;
}

