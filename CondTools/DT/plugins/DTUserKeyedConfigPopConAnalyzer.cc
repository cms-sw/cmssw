#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTUserKeyedConfigHandler.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondCore/CondDB/interface/KeyList.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"
#include "CondFormats/DataRecord/interface/DTKeyedConfigListRcd.h"
#include <memory>

//typedef popcon::PopConAnalyzer<DTUserKeyedConfigHandler> DTUserKeyedConfigPopConAnalyzer;
class DTUserKeyedConfigPopConAnalyzer : public popcon::PopConAnalyzer<DTUserKeyedConfigHandler> {
public:
  DTUserKeyedConfigPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<DTUserKeyedConfigHandler>(pset) {}
  ~DTUserKeyedConfigPopConAnalyzer() override {}
  void analyze(const edm::Event& e, const edm::EventSetup& s) override {
    edm::ESHandle<cond::persistency::KeyList> klh;
    std::cout << "got eshandle" << std::endl;
    s.get<DTKeyedConfigListRcd>().get(klh);
    std::cout << "got context" << std::endl;
    cond::persistency::KeyList const& kl = *klh.product();
    for (size_t i = 0; i < kl.size(); i++) {
      std::shared_ptr<DTKeyedConfig> kentry = kl.getUsingIndex<DTKeyedConfig>(i);
      if (kentry.get())
        std::cout << kentry->getId() << std::endl;
    }
    source().setList(&kl);
  }

private:
};

DEFINE_FWK_MODULE(DTUserKeyedConfigPopConAnalyzer);
