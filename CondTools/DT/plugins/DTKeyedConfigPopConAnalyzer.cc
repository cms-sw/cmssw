#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTKeyedConfigHandler.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondCore/CondDB/interface/KeyList.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"
#include "CondFormats/DataRecord/interface/DTKeyedConfigListRcd.h"
#include <memory>

//typedef popcon::PopConAnalyzer<DTKeyedConfigHandler> DTKeyedConfigPopConAnalyzer;
class DTKeyedConfigPopConAnalyzer : public popcon::PopConAnalyzer<DTKeyedConfigHandler> {
public:
  DTKeyedConfigPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<DTKeyedConfigHandler>(pset),
        copyData(pset.getParameter<edm::ParameterSet>("Source").getUntrackedParameter<bool>("copyData", true)),
        perskeylistToken_(esConsumes()) {}
  ~DTKeyedConfigPopConAnalyzer() override {}
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    if (!copyData)
      return;
    edm::LogInfo("DTKeyedConfigPopConAnalyzer") << "got context" << std::endl;
    cond::persistency::KeyList const& kl = iSetup.getData(perskeylistToken_);
    for (size_t i = 0; i < kl.size(); i++) {
      std::shared_ptr<DTKeyedConfig> kelem = kl.getUsingIndex<DTKeyedConfig>(i);
      if (kelem.get())
        edm::LogInfo("DTKeyedConfigPopConAnalyzer") << kelem->getId() << std::endl;
    }
    source().setList(&kl);
  }

private:
  bool copyData;
  edm::ESGetToken<cond::persistency::KeyList, DTKeyedConfigListRcd> perskeylistToken_;
};

DEFINE_FWK_MODULE(DTKeyedConfigPopConAnalyzer);
