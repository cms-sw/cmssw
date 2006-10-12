
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include <CalibCalorimetry/HcalStandardModules/interface/HcalPedestalAnalyzer.h>
#include "CalibCalorimetry/HcalAlgos/interface/HcalAlgoUtils.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondTools/Hcal/interface/HcalDbPool.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"

#include <iostream>
#include <fstream>

/*
 * \file HcalPedestalAnalyzer.cc
 * 
 * $Date: 2006/01/14 00:42:12 $
 * $Revision: 1.1 $
 * \author S Stoynev / W Fisher
 *
*/

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
      std::cout << "HcalPedestalAnalyzer-> USE INPUT: ASCII " << std::endl;
      std::ifstream stream (fDb.c_str ());
      HcalDbASCIIIO::getObject (stream, fObject); 
      return true;
    }
    else if (dbFile (fDb)) {
      std::cout << "HcalPedestalAnalyzer-> USE INPUT: Pool " << fDb << std::endl;
      HcalDbPool poolDb (fDb);
      return poolDb.getObject (fObject, fTag, fRun);
    }
    else {
      std::cerr << "HcalPedestalAnalyzer-> Unknown input type " << fDb << std::endl;
      return false;
    }
  }
  
  template <class T> 
  bool putObject (T** fObject, const std::string& fDb, const std::string& fTag, int fRun) {
    if (!fObject || !*fObject) return false;
    if (fDb.empty ()) return false; 
    if (asciiFile (fDb)) {
      std::cout << "HcalPedestalAnalyzer-> USE OUTPUT: ASCII " << std::endl;
      std::ofstream stream (fDb.c_str ());
      HcalDbASCIIIO::dumpObject (stream, **fObject); 
      return true;
    }
//  XML will need more parameters - bypass it for now
//     else if (xmlFile (fDb)) {
//       std::cout << "HcalPedestalAnalyzer-> USE OUTPUT: XML" << std::endl;
// 	std::ofstream stream (fDb.c_str ());
// 	HcalDbXml::dumpObject (stream, fRun, fIovgmtbegin, fIovgmtend, fOutputTag, fVersion, *object);
// 	stream.close ();
//       }
    else if (dbFile (fDb)) {
      std::cout << "HcalPedestalAnalyzer-> USE OUTPUT: Pool " << fDb << std::endl;
      HcalDbPool poolDb (fDb);
      bool result = poolDb.putObject (*fObject, fTag, fRun);
      if (result) *fObject = 0; // owned by POOL
      return result;
    }
    else {
      std::cerr << "HcalPedestalAnalyzer-> Unknown output type " << fDb << std::endl;
      return false;
    }
  }
}

HcalPedestalAnalyzer::HcalPedestalAnalyzer(const edm::ParameterSet& ps){

  m_pedAnal = new HcalPedestalAnalysis(ps);
  m_pedAnal->setup(ps.getUntrackedParameter<std::string>("outputFileHist", "HcalPedestalAnalyzer.root"));

  m_startSample = ps.getUntrackedParameter<int>("firstSample", 0);
  m_endSample = ps.getUntrackedParameter<int>("lastSample", 19);
  m_inputPedestals_source = ps.getUntrackedParameter<std::string>("inputPedestalsSource", "");
  m_inputPedestals_tag = ps.getUntrackedParameter<std::string>("inputPedestalsTag", "");
  m_inputPedestals_run = ps.getUntrackedParameter<int>("inputPedestalsRun", 1);
  m_inputPedestalWidths_source = ps.getUntrackedParameter<std::string>("inputPedestalWidthsSource", "");
  m_inputPedestalWidths_tag = ps.getUntrackedParameter<std::string>("inputPedestalWidthsTag", "");
  m_inputPedestalWidths_run = ps.getUntrackedParameter<int>("inputPedestalWidthsRun", 1);
  m_outputPedestals_dest = ps.getUntrackedParameter<std::string>("outputPedestalsDest", "");
  m_outputPedestals_tag = ps.getUntrackedParameter<std::string>("outputPedestalsTag", "");
  m_outputPedestals_run = ps.getUntrackedParameter<int>("outputPedestalsRun", 99999);
  m_outputPedestalWidths_dest = ps.getUntrackedParameter<std::string>("outputPedestalWidthsDest", "");
  m_outputPedestalWidths_tag = ps.getUntrackedParameter<std::string>("outputPedestalWidthsTag", "");
  m_outputPedestalWidths_run = ps.getUntrackedParameter<int>("outputPedestalWidthsRun", 99999);

  // CORAL required variables to be set, even if not needed
  if (!::getenv("CORAL_AUTH_USER")) ::putenv("CORAL_AUTH_USER=blah");
  if (!::getenv("CORAL_AUTH_PASSWORD")) ::putenv("CORAL_AUTH_PASSWORD=blah"); 
}

HcalPedestalAnalyzer::~HcalPedestalAnalyzer(){
  delete m_pedAnal;
}

void HcalPedestalAnalyzer::beginJob(const edm::EventSetup& c){
  m_ievt = 0;
  ped_sample = 1;
}

void HcalPedestalAnalyzer::endJob(void) {
  // get input objects
  HcalPedestals* inputPeds = 0;
  if (!m_inputPedestals_source.empty ()) {
    inputPeds = new HcalPedestals ();
    if (!getObject (inputPeds, m_inputPedestals_source, m_inputPedestals_tag, m_inputPedestals_run)) {
      std::cerr << "HcalPedestalAnalyzer-> Failed to get input Pedestals" << std::endl;
    }
  }
  HcalPedestalWidths* inputPedWids = 0;
  if (!m_inputPedestalWidths_source.empty ()) {
    inputPedWids = new HcalPedestalWidths ();
    if (!getObject (inputPedWids, m_inputPedestalWidths_source, m_inputPedestalWidths_tag, m_inputPedestalWidths_run)) {
      std::cerr << "HcalPedestalAnalyzer-> Failed to get input PedestalWidths" << std::endl;
    }
  }

  // make output objects
  HcalPedestals* outputPeds = m_outputPedestals_dest.empty () ? 0 : new HcalPedestals ();
  HcalPedestalWidths* outputPedWids = m_outputPedestalWidths_dest.empty () ? 0 : new HcalPedestalWidths ();

  // run algorithm
  m_pedAnal->done(inputPeds, inputPedWids, outputPeds, outputPedWids);

  delete inputPeds;
  delete inputPedWids;

  // store new objects
  if (outputPeds) {
    if (!putObject (&outputPeds, m_outputPedestals_dest, m_outputPedestals_tag, m_outputPedestals_run)) {
      std::cerr << "HcalPedestalAnalyzer-> Failed to put output Pedestals" << std::endl;
    }
    delete outputPeds;
  }
  if (outputPedWids) {
    if (!putObject (&outputPedWids, m_outputPedestalWidths_dest, m_outputPedestalWidths_tag, m_outputPedestalWidths_run)) {
      std::cerr << "HcalPedestalAnalyzer-> Failed to put output PedestalWidths" << std::endl;
    }
    delete outputPedWids;
  }
}

void HcalPedestalAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& eventSetup){

  m_ievt++;

  ///get digis
  edm::Handle<HBHEDigiCollection> hbhe; e.getByType(hbhe);
  edm::Handle<HODigiCollection> ho;     e.getByType(ho);
  edm::Handle<HFDigiCollection> hf;     e.getByType(hf);

  // get conditions
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);

  m_pedAnal->processEvent(*hbhe, *ho, *hf, *conditions);

  if(m_ievt%1000 == 0)
    std::cout << "HcalPedestalAnalyzer: analyzed " << m_ievt << " events" << std::endl;

  return;
}

// #include "PluginManager/ModuleDef.h"
// #include "FWCore/Framework/interface/MakerMacros.h"

// DEFINE_SEAL_MODULE();
// DEFINE_ANOTHER_FWK_MODULE(HcalPedestalAnalyzer)
