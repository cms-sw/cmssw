#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class scaleGains : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit scaleGains(const edm::ParameterSet&);
  ~scaleGains() override;

private:
  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
  std::string fileIn, fileOut;
  double scale;
};

scaleGains::scaleGains(const edm::ParameterSet& iConfig) {
  fileIn = iConfig.getUntrackedParameter<std::string>("FileIn");
  fileOut = iConfig.getUntrackedParameter<std::string>("FileOut");
  scale = iConfig.getUntrackedParameter<double>("Scale");
}

scaleGains::~scaleGains() {}

void scaleGains::analyze(edm::Event const&, edm::EventSetup const& iSetup) {
  edm::ESHandle<HcalTopology> htopo;
  iSetup.get<HcalRecNumberingRecord>().get(htopo);
  HcalTopology topo = (*htopo);

  HcalGains gainsIn(&topo);
  ;
  std::ifstream inStream(fileIn.c_str());
  HcalDbASCIIIO::getObject(inStream, &gainsIn);
  inStream.close();

  HcalGains gainsOut(&topo);
  ;
  std::vector<DetId> channels = gainsIn.getAllChannels();
  for (unsigned i = 0; i < channels.size(); i++) {
    DetId id = channels[i];
    HcalGain item(id,
                  gainsIn.getValues(id)->getValue(0) * scale,
                  gainsIn.getValues(id)->getValue(1) * scale,
                  gainsIn.getValues(id)->getValue(2) * scale,
                  gainsIn.getValues(id)->getValue(3) * scale);
    gainsOut.addValues(item);
  }

  std::ofstream outStream(fileOut.c_str());
  HcalDbASCIIIO::dumpObject(outStream, gainsOut);
  outStream.close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(scaleGains);
