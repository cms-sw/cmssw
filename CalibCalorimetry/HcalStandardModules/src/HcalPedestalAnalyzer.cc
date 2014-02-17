
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include <CalibCalorimetry/HcalStandardModules/interface/HcalPedestalAnalyzer.h>
//#include "CondTools/Hcal/interface/HcalDbTool.h"
#include "CondTools/Hcal/interface/HcalDbOnline.h"
#include "CondTools/Hcal/interface/HcalDbXml.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"


/*
 * \file HcalPedestalAnalyzer.cc
 * 
 * $Date: 2012/11/13 03:30:20 $
 * $Revision: 1.16 $
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

  bool masterDb (const std::string fParam) {
    return fParam.find ('@') != std::string::npos;
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
      std::cout << "HcalPedestalAnalyzer-> Pool interface is not supportet since 1.3.0" << fDb << std::endl;
      return false;
//       HcalDbTool poolDb (fDb);
//       return poolDb.getObject (fObject, fTag, fRun);
    }
    else if (masterDb (fDb)) {
      std::cout << "HcalPedestalAnalyzer-> USE INPUT: MasterDB " << fDb << std::endl;
      HcalDbOnline masterDb (fDb);
      return masterDb.getObject (fObject, fTag, fRun);
    }
    else {
      std::cerr << "HcalPedestalAnalyzer-> Unknown input type " << fDb << std::endl;
      return false;
    }
  }
  
  bool dumpXmlPedestals (const HcalPedestals& fObject, const HcalPedestalWidths& fWidth, const std::string& fXml, const std::string& fTag, int fRun) {
    std::cout << "HcalPedestalAnalyzer-> USE OUTPUT: XML" << std::endl;
    std::ofstream stream (fXml.c_str ());
    bool result = HcalDbXml::dumpObject (stream, fRun, fRun, 0, fTag, fObject, fWidth);
    stream.close ();
    return result;
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
    else if (dbFile (fDb)) {
      std::cout << "HcalPedestalAnalyzer-> USE OUTPUT: Pool " << fDb << std::endl;
      std::cout << "HcalPedestalAnalyzer-> Pool interface is not supportet since 1.3.0" << fDb << std::endl;
      return false;
//       HcalDbTool poolDb (fDb);
//       bool result = poolDb.putObject (*fObject, fTag, fRun);
//       if (result) *fObject = 0; // owned by POOL
//       return result;
    }
    else {
      std::cerr << "HcalPedestalAnalyzer-> Unknown output type " << fDb << std::endl;
      return false;
    }
  }
}

HcalPedestalAnalyzer::HcalPedestalAnalyzer(const edm::ParameterSet& ps) :
  hbheDigiCollectionTag_(ps.getParameter<edm::InputTag>("hbheDigiCollectionTag")),
  hoDigiCollectionTag_(ps.getParameter<edm::InputTag>("hoDigiCollectionTag")),
  hfDigiCollectionTag_(ps.getParameter<edm::InputTag>("hfDigiCollectionTag")) {

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
  const char* foo1 = "CORAL_AUTH_USER=blaaah";
  const char* foo2 = "CORAL_AUTH_PASSWORD=blaaah";
  if (!::getenv("CORAL_AUTH_USER")) ::putenv(const_cast<char*>(foo1));
  if (!::getenv("CORAL_AUTH_PASSWORD")) ::putenv(const_cast<char*>(foo2)); 
}

HcalPedestalAnalyzer::~HcalPedestalAnalyzer(){
//  delete m_pedAnal;
}

void HcalPedestalAnalyzer::beginJob(){
  m_ievt = 0;
  ped_sample = 1;
}

void HcalPedestalAnalyzer::endJob(void) {
  // get input objects
  HcalPedestals* inputPeds = 0;
  if (!m_inputPedestals_source.empty ()) {
    inputPeds = new HcalPedestals (m_topo);
    if (!getObject (inputPeds, m_inputPedestals_source, m_inputPedestals_tag, m_inputPedestals_run)) {
      std::cerr << "HcalPedestalAnalyzer-> Failed to get input Pedestals" << std::endl;
    }
  }
  HcalPedestalWidths* inputPedWids = 0;
  if (!m_inputPedestalWidths_source.empty ()) {
    inputPedWids = new HcalPedestalWidths (m_topo);
    if (!getObject (inputPedWids, m_inputPedestalWidths_source, m_inputPedestalWidths_tag, m_inputPedestalWidths_run)) {
      std::cerr << "HcalPedestalAnalyzer-> Failed to get input PedestalWidths" << std::endl;
    }
  }

  // make output objects
  HcalPedestals* outputPeds = (m_outputPedestals_dest.empty () && !xmlFile (m_outputPedestals_dest)) ? 0 : new HcalPedestals (m_topo);
  HcalPedestalWidths* outputPedWids = (m_outputPedestalWidths_dest.empty () && !xmlFile (m_outputPedestals_dest)) ? 0 : new HcalPedestalWidths (m_topo);

  // run algorithm
  int Flag=m_pedAnal->done(inputPeds, inputPedWids, outputPeds, outputPedWids);

  delete inputPeds;
  delete inputPedWids;


  // store new objects
  // Flag=-2 indicates there were less than 100 events and output is meaningless
  if (Flag>-2) {
    if (xmlFile (m_outputPedestals_dest)) { // output pedestals and widths together
      if (!dumpXmlPedestals (*outputPeds, *outputPedWids, m_outputPedestals_dest, m_outputPedestals_tag, m_outputPedestals_run)) {
	std::cerr << "HcalPedestalAnalyzer-> Failed to put output Pedestals & Widths" << std::endl;
      }
    }
    else {
      if (outputPeds) {
	if (!putObject (&outputPeds, m_outputPedestals_dest, m_outputPedestals_tag, m_outputPedestals_run)) {
	  std::cerr << "HcalPedestalAnalyzer-> Failed to put output Pedestals" << std::endl;
	}
      }
      if (outputPedWids) {
	if (!putObject (&outputPedWids, m_outputPedestalWidths_dest, m_outputPedestalWidths_tag, m_outputPedestalWidths_run)) {
	  std::cerr << "HcalPedestalAnalyzer-> Failed to put output PedestalWidths" << std::endl;
	}
      }
    }
  }
  delete outputPeds;
  delete outputPedWids;
}

void HcalPedestalAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& eventSetup){

  m_ievt++;

  ///get digis
  edm::Handle<HBHEDigiCollection> hbhe; e.getByLabel(hbheDigiCollectionTag_, hbhe);
  edm::Handle<HODigiCollection> ho;     e.getByLabel(hoDigiCollectionTag_, ho);
  edm::Handle<HFDigiCollection> hf;     e.getByLabel(hfDigiCollectionTag_, hf);

  // get conditions
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);

  if (m_topo==0) {
    const IdealGeometryRecord& record = eventSetup.get<IdealGeometryRecord>();
    edm::ESHandle<HcalTopology> topology;
    record.get (topology);
    m_topo=new HcalTopology(*topology);
  }
  

  m_pedAnal->processEvent(*hbhe, *ho, *hf, *conditions);

  if(m_ievt%1000 == 0)
    std::cout << "HcalPedestalAnalyzer: analyzed " << m_ievt << " events" << std::endl;

  return;
}

// #include "FWCore/PluginManager/interface/ModuleDef.h"
// #include "FWCore/Framework/interface/MakerMacros.h"

// 
// DEFINE_FWK_MODULE(HcalPedestalAnalyzer);
