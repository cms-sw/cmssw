#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalChannelQualityHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalChannelQualityHandler> HcalChannelQualityPopConAnalyzer;

class HcalChannelQualityPopConAnalyzer : public popcon::PopConAnalyzer<HcalChannelQualityHandler> {
public:
  typedef HcalChannelQualityHandler SourceHandler;

  HcalChannelQualityPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalChannelQualityHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalChannelQuality, HcalChannelQualityRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalChannelQuality(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalChannelQuality, HcalChannelQualityRcd> m_tok;

  HcalChannelQuality* myDBObject;
};

DEFINE_FWK_MODULE(HcalChannelQualityPopConAnalyzer);
