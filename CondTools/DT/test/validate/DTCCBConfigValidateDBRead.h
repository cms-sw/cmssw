/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
class DTConfigKey;
class DTCCBConfig;
class DTCCBConfigRcd;
class DTCCBConfigValidateDBRead : public edm::one::EDAnalyzer<> {
public:
  explicit DTCCBConfigValidateDBRead(edm::ParameterSet const& p);
  explicit DTCCBConfigValidateDBRead(int i);
  ~DTCCBConfigValidateDBRead() override = default;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void endJob() override;

private:
  std::string dataFileName;
  std::string elogFileName;
  edm::ESGetToken<DTCCBConfig, DTCCBConfigRcd> dtccbToken_;
  static bool cfrDiff(const std::vector<int>& l_conf, const std::vector<int>& r_conf);
  static bool cfrDiff(const std::vector<DTConfigKey>& l_conf, const std::vector<DTConfigKey>& r_conf);
};
