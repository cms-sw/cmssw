//G.Benelli Aug 26 2010
//Plugin adapted from Andrea Venturi's myTkAnalyses/PSTools to create Voltage (LV/HV) ON/OFF Tracke maps in the context of Strip DCS O2O.

// system include files
#include <memory>

// user include files

#include <iostream>
#include <fstream>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

//
// class decleration
//

class TkVoltageMapCreator : public edm::EDAnalyzer {
public:
  explicit TkVoltageMapCreator(const edm::ParameterSet&);
  ~TkVoltageMapCreator() override;

private:
  void beginJob() override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------

  const std::string _lvfile;
  const std::string _lvtkmapname;
  const std::string _hvfile;
  const std::string _hvtkmapname;

  edm::ESGetToken<TkDetMap, TrackerTopologyRcd> tkDetMapToken_;
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
TkVoltageMapCreator::TkVoltageMapCreator(const edm::ParameterSet& iConfig)
    : _lvfile(iConfig.getParameter<std::string>("LVStatusFile")),
      _lvtkmapname(iConfig.getParameter<std::string>("LVTkMapName")),
      _hvfile(iConfig.getParameter<std::string>("HVStatusFile")),
      _hvtkmapname(iConfig.getParameter<std::string>("HVTkMapName")),
      tkDetMapToken_(esConsumes<edm::Transition::BeginRun>())

{
  //now do what ever initialization is needed
}

TkVoltageMapCreator::~TkVoltageMapCreator() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void TkVoltageMapCreator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) { using namespace edm; }

void TkVoltageMapCreator::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  const TkDetMap* tkDetMap = &iSetup.getData(tkDetMapToken_);

  TrackerMap lvmap, hvmap;

  TkHistoMap lvhisto(tkDetMap, "LV_Status", "LV_Status", -1);
  TkHistoMap hvhisto(tkDetMap, "HV_Status", "HV_Status", -1);

  std::ifstream lvdata(_lvfile.c_str());
  std::ifstream hvdata(_hvfile.c_str());

  // HV channel map filling

  unsigned int detid;
  std::string lvstatus;

  while (lvdata >> detid >> lvstatus) {
    double cha = 0.;
    if (lvstatus == "ON")
      cha = 0.5;  //GREEN
    if (lvstatus == "OFF")
      cha = 1.;  //RED
    lvhisto.fill(detid, cha);
  }

  std::string hvstatus;

  while (hvdata >> detid >> hvstatus) {
    double cha = 0.;
    if (hvstatus == "ON")
      cha = 0.5;  //GREEN
    if (hvstatus == "OFF")
      cha = 1.;  //RED
    hvhisto.fill(detid, cha);
  }

  lvmap.setPalette(1);
  hvmap.setPalette(1);

  lvhisto.dumpInTkMap(&lvmap);
  hvhisto.dumpInTkMap(&hvmap);

  lvmap.save(true, 0, 0, _lvtkmapname);
  hvmap.save(true, 0, 0, _hvtkmapname);

  //TODO could make the root file name a parameter to avoid overwriting everytime...
  std::string rootmapname = "VoltageStatus.root";
  lvhisto.save(rootmapname);
  hvhisto.save(rootmapname);
}

void TkVoltageMapCreator::endRun(const edm::Run& iRun, const edm::EventSetup&) {}

// ------------ method called once each job just before starting event loop  ------------
void TkVoltageMapCreator::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void TkVoltageMapCreator::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(TkVoltageMapCreator);
