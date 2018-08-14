#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class corrResps : public edm::one::EDAnalyzer<edm::one::WatchRuns> {

public:
  explicit corrResps(const edm::ParameterSet&);
  ~corrResps() override;

private:
  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
  std::string fileIn, fileOut, fileCorr;
};

corrResps::corrResps(const edm::ParameterSet& iConfig) {
  fileIn   = iConfig.getUntrackedParameter<std::string>("FileIn");
  fileOut  = iConfig.getUntrackedParameter<std::string>("FileOut");
  fileCorr = iConfig.getUntrackedParameter<std::string>("FileCorr");
}

corrResps::~corrResps() { }

void corrResps::analyze(edm::Event const&, edm::EventSetup const& iSetup) {

  edm::ESHandle<HcalTopology> htopo;
  iSetup.get<HcalRecNumberingRecord>().get(htopo);
  HcalTopology topo = (*htopo);

  HcalRespCorrs respIn(&topo);
  std::ifstream inStream  (fileIn.c_str());
  HcalDbASCIIIO::getObject (inStream, &respIn);
  inStream.close();

  HcalRespCorrs corrsIn(&topo);
  std::ifstream inCorr     (fileCorr.c_str());
  HcalDbASCIIIO::getObject (inCorr, &corrsIn);
  inCorr.close();

  HcalRespCorrs respOut(&topo);
  std::vector<DetId> channels = respIn.getAllChannels ();
  for (unsigned int i = 0; i < channels.size(); i++) {
    DetId id = channels[i];
    float scale = 1.0;
    if (corrsIn.exists(id)) scale = corrsIn.getValues(id)->getValue();
    HcalRespCorr item (id, respIn.getValues(id)->getValue() * scale);
    respOut.addValues(item);
  }

  std::ofstream outStream (fileOut.c_str());
  HcalDbASCIIIO::dumpObject (outStream, respOut);
  outStream.close();
}


//define this as a plug-in
DEFINE_FWK_MODULE(corrResps);
