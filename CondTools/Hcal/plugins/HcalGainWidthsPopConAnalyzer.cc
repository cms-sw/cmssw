#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalGainWidthsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalGainWidthsHandler> HcalGainWidthsPopConAnalyzer;

class HcalGainWidthsPopConAnalyzer : public popcon::PopConAnalyzer<HcalGainWidthsHandler> {
public:
  typedef HcalGainWidthsHandler SourceHandler;

  HcalGainWidthsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalGainWidthsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalGainWidths, HcalGainWidthsRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalGainWidths(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalGainWidths, HcalGainWidthsRcd> m_tok;

  HcalGainWidths* myDBObject;
};

DEFINE_FWK_MODULE(HcalGainWidthsPopConAnalyzer);
