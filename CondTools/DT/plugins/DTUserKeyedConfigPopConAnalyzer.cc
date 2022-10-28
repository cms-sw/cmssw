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
      : popcon::PopConAnalyzer<DTUserKeyedConfigHandler>(pset), perskeylistToken_(esConsumes()) {}
  ~DTUserKeyedConfigPopConAnalyzer() override {}
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    edm::LogInfo("DTUserKeyedConfigPopConAnalyzer") << "got eSdata" << std::endl;
    cond::persistency::KeyList const& kl = iSetup.getData(perskeylistToken_);
    for (size_t i = 0; i < kl.size(); i++) {
      std::shared_ptr<DTKeyedConfig> kentry = kl.getUsingIndex<DTKeyedConfig>(i);
      if (kentry.get())
        edm::LogInfo("DTUserKeyedConfigPopConAnalyzer") << kentry->getId() << std::endl;
    }
    source().setList(&kl);
  }

private:
  edm::ESGetToken<cond::persistency::KeyList, DTKeyedConfigListRcd> perskeylistToken_;
};

DEFINE_FWK_MODULE(DTUserKeyedConfigPopConAnalyzer);
