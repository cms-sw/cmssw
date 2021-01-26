#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalZSThresholdsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalZSThresholdsHandler> HcalZSThresholdsPopConAnalyzer;

class HcalZSThresholdsPopConAnalyzer : public popcon::PopConAnalyzer<HcalZSThresholdsHandler> {
public:
  typedef HcalZSThresholdsHandler SourceHandler;

  HcalZSThresholdsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalZSThresholdsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalZSThresholds, HcalZSThresholdsRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalZSThresholds(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalZSThresholds, HcalZSThresholdsRcd> m_tok;

  HcalZSThresholds* myDBObject;
};

DEFINE_FWK_MODULE(HcalZSThresholdsPopConAnalyzer);
