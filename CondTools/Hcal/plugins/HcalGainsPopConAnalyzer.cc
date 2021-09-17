#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalGainsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalGainsHandler> HcalGainsPopConAnalyzer;

class HcalGainsPopConAnalyzer : public popcon::PopConAnalyzer<HcalGainsHandler> {
public:
  typedef HcalGainsHandler SourceHandler;

  HcalGainsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalGainsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalGains, HcalGainsRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalGains(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalGains, HcalGainsRcd> m_tok;

  HcalGains* myDBObject;
};

DEFINE_FWK_MODULE(HcalGainsPopConAnalyzer);
