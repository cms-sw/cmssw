#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalDcsValuesHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalDcsValuesHandler> HcalDcsValuesPopConAnalyzer;

class HcalDcsValuesPopConAnalyzer : public popcon::PopConAnalyzer<HcalDcsValuesHandler> {
public:
  typedef HcalDcsValuesHandler SourceHandler;

  HcalDcsValuesPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalDcsValuesHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalDcsValues, HcalDcsRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalDcsValues(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalDcsValues, HcalDcsRcd> m_tok;

  HcalDcsValues* myDBObject;
};

DEFINE_FWK_MODULE(HcalDcsValuesPopConAnalyzer);
