#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalFrontEndMapHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class HcalFrontEndMapPopConAnalyzer : public popcon::PopConAnalyzer<HcalFrontEndMapHandler> {
public:
  typedef HcalFrontEndMapHandler SourceHandler;

  HcalFrontEndMapPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalFrontEndMapHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalFrontEndMap, HcalFrontEndMapRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalFrontEndMap(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalFrontEndMap, HcalFrontEndMapRcd> m_tok;

  HcalFrontEndMap* myDBObject;
};

DEFINE_FWK_MODULE(HcalFrontEndMapPopConAnalyzer);
