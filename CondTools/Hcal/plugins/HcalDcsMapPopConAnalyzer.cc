#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalDcsMapHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalDcsMapHandler> HcalDcsMapPopConAnalyzer;

class HcalDcsMapPopConAnalyzer : public popcon::PopConAnalyzer<HcalDcsMapHandler> {
public:
  typedef HcalDcsMapHandler SourceHandler;

  HcalDcsMapPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalDcsMapHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalDcsMap, HcalDcsMapRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalDcsMap(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalDcsMap, HcalDcsMapRcd> m_tok;

  HcalDcsMap* myDBObject;
};

DEFINE_FWK_MODULE(HcalDcsMapPopConAnalyzer);
