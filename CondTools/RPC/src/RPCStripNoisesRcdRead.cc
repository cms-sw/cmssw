#include <string>
#include <map>
#include <vector>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondTools/RPC/interface/RPCDBSimSetUp.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
#include "CondFormats/DataRecord/interface/RPCStripNoisesRcd.h"

class RPCStripNoisesRcdRead : public edm::one::EDAnalyzer<> {
public:
  RPCStripNoisesRcdRead(const edm::ParameterSet& iConfig);
  ~RPCStripNoisesRcdRead() override;
  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;

private:
  edm::ESGetToken<RPCStripNoises, RPCStripNoisesRcd> noise_RcdToken_;
};

RPCStripNoisesRcdRead::RPCStripNoisesRcdRead(const edm::ParameterSet& iConfig) : noise_RcdToken_(esConsumes()) {}

RPCStripNoisesRcdRead::~RPCStripNoisesRcdRead() {}

void RPCStripNoisesRcdRead::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  const RPCStripNoises* noiseRcd = &evtSetup.getData(noise_RcdToken_);
  edm::LogInfo("RPCStripNoisesReader") << "[RPCStripNoisesReader::analyze] End Reading RPCStripNoises" << std::endl;
  std::vector<RPCStripNoises::NoiseItem> vnoise = noiseRcd->getVNoise();
  std::vector<float> vcls = noiseRcd->getCls();

  for (unsigned int n = 0; n < vcls.size(); ++n) {
    std::cout << "Cls Value: " << vcls[n] << std::endl;
  }

  int i = 1;
  for (std::vector<RPCStripNoises::NoiseItem>::iterator it = vnoise.begin(); it != vnoise.end(); ++it) {
    if (i % 96 == 0)
      std::cout << "DetId:  " << it->dpid << "  " << it->time << "  " << std::endl;
    std::cout << "                                    Noise Value: " << (it->noise) << "  " << (it->eff) << std::endl;
    i++;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCStripNoisesRcdRead);
