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
        copyData(pset.getParameter<edm::ParameterSet>("Source").getUntrackedParameter<bool>("copyData", true)) {}
  ~DTKeyedConfigPopConAnalyzer() override {}
  void analyze(const edm::Event& e, const edm::EventSetup& s) override {
    if (!copyData)
      return;

    edm::ESHandle<cond::persistency::KeyList> klh;
    std::cout << "got eshandle" << std::endl;
    s.get<DTKeyedConfigListRcd>().get(klh);
    std::cout << "got context" << std::endl;
    cond::persistency::KeyList const& kl = *klh.product();
    for (size_t i = 0; i < kl.size(); i++) {
      std::shared_ptr<DTKeyedConfig> kelem = kl.getUsingIndex<DTKeyedConfig>(i);
      if (kelem.get())
        std::cout << kelem->getId() << std::endl;
    }
    source().setList(&kl);
  }

private:
  bool copyData;
};

DEFINE_FWK_MODULE(DTKeyedConfigPopConAnalyzer);
