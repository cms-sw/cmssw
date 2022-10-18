#include <sys/time.h>
#include <stdint.h>
#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"

#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

using namespace PhysicsTools;

class testReadMVAComputerCondDB : public edm::one::EDAnalyzer<> {
public:
  explicit testReadMVAComputerCondDB(const edm::ParameterSet& params);

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  void endJob() override;

private:
  edm::ESGetToken<Calibration::MVAComputerContainer, BTauGenericMVAJetTagComputerRcd> m_token;
};

testReadMVAComputerCondDB::testReadMVAComputerCondDB(const edm::ParameterSet& params) : m_token(esConsumes()) {}

void testReadMVAComputerCondDB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  MVAComputer computer(&iSetup.getData(m_token).find("test"));

  Variable::Value values[] = {
      Variable::Value("toast", 4.4), Variable::Value("toast", 4.5), Variable::Value("test", 4.6),
      Variable::Value("toast", 4.7), Variable::Value("test", 4.8),  Variable::Value("normal", 4.9),
      Variable::Value("toast", 4.4), Variable::Value("toast", 4.5), Variable::Value("test", 4.6),
      Variable::Value("toast", 4.7), Variable::Value("test", 4.8),  Variable::Value("normal", 4.9),
      Variable::Value("toast", 4.4), Variable::Value("toast", 4.5), Variable::Value("test", 4.6),
      Variable::Value("toast", 4.7), Variable::Value("test", 4.8),  Variable::Value("normal", 4.9),
      Variable::Value("toast", 4.4), Variable::Value("toast", 4.5), Variable::Value("test", 4.6),
      Variable::Value("toast", 4.7), Variable::Value("test", 4.8),  Variable::Value("normal", 4.9)};

  unsigned int i = 0;
  uint64_t n = 0;
  struct timeval start;
  gettimeofday(&start, 0);
  for (;;) {
    computer.eval(values, values + 6);
    n++;
    if (++i == 1000) {
      i = 0;
      struct timeval now;
      gettimeofday(&now, NULL);
      if (now.tv_sec < start.tv_sec + 5)
        continue;
      if (now.tv_sec > start.tv_sec + 5)
        break;
      if (now.tv_usec >= start.tv_usec)
        break;
    }
  }

  std::cout << "Did " << n << " computationss in five seconds." << std::endl;
}

void testReadMVAComputerCondDB::endJob() {}

// define this as a plug-in
DEFINE_FWK_MODULE(testReadMVAComputerCondDB);
