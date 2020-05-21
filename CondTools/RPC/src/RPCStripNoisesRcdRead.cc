#include <string>
#include <map>
#include <vector>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondTools/RPC/interface/RPCDBSimSetUp.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
#include "CondFormats/DataRecord/interface/RPCStripNoisesRcd.h"

class RPCStripNoisesRcdRead : public edm::EDAnalyzer {
public:
  RPCStripNoisesRcdRead(const edm::ParameterSet& iConfig);
  ~RPCStripNoisesRcdRead() override;
  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;
};

RPCStripNoisesRcdRead::RPCStripNoisesRcdRead(const edm::ParameterSet& iConfig) {}

RPCStripNoisesRcdRead::~RPCStripNoisesRcdRead() {}

void RPCStripNoisesRcdRead::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  edm::ESHandle<RPCStripNoises> noiseRcd;
  evtSetup.get<RPCStripNoisesRcd>().get(noiseRcd);
  edm::LogInfo("RPCStripNoisesReader") << "[RPCStripNoisesReader::analyze] End Reading RPCStripNoises" << std::endl;

  std::vector<RPCStripNoises::NoiseItem> vnoise = noiseRcd->getVNoise();
  std::vector<float> vcls = noiseRcd->getCls();

  for (float vcl : vcls) {
    std::cout << "Cls Value: " << vcl << std::endl;
  }

  int i = 1;
  for (auto& it : vnoise) {
    if (i % 96 == 0)
      std::cout << "DetId:  " << it.dpid << "  " << it.time << "  " << std::endl;
    std::cout << "                                    Noise Value: " << (it.noise) << "  " << (it.eff) << std::endl;
    i++;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCStripNoisesRcdRead);
