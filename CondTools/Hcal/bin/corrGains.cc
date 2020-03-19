#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class corrGains : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit corrGains(const edm::ParameterSet&);
  ~corrGains() override;

private:
  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
  std::string fileIn, fileOut, fileCorr;
};

corrGains::corrGains(const edm::ParameterSet& iConfig) {
  fileIn = iConfig.getUntrackedParameter<std::string>("FileIn");
  fileOut = iConfig.getUntrackedParameter<std::string>("FileOut");
  fileCorr = iConfig.getUntrackedParameter<std::string>("FileCorr");
}

corrGains::~corrGains() {}

void corrGains::analyze(edm::Event const&, edm::EventSetup const& iSetup) {
  edm::ESHandle<HcalTopology> htopo;
  iSetup.get<HcalRecNumberingRecord>().get(htopo);
  HcalTopology topo = (*htopo);

  HcalGains gainsIn(&topo);
  ;
  std::ifstream inStream(fileIn.c_str());
  HcalDbASCIIIO::getObject(inStream, &gainsIn);
  inStream.close();

  HcalRespCorrs corrsIn(&topo);
  ;
  std::ifstream inCorr(fileCorr.c_str());
  HcalDbASCIIIO::getObject(inCorr, &corrsIn);
  inCorr.close();

  HcalGains gainsOut(&topo);
  ;
  std::vector<DetId> channels = gainsIn.getAllChannels();
  for (unsigned int i = 0; i < channels.size(); i++) {
    DetId id = channels[i];
    float scale = 1.;
    if (corrsIn.exists(id))
      scale = corrsIn.getValues(id)->getValue();
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
DEFINE_FWK_MODULE(corrGains);
