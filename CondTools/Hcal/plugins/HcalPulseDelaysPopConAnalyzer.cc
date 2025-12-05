#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondFormats/DataRecord/interface/HcalPulseDelaysRcd.h"
#include "CondTools/Hcal/interface/HcalPulseDelaysHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class HcalPulseDelaysPopConAnalyzer : public popcon::PopConAnalyzer<HcalPulseDelaysHandler> {
public:
  typedef HcalPulseDelaysHandler SourceHandler;

  HcalPulseDelaysPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalPulseDelaysHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalPulseDelays, HcalPulseDelaysRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalPulseDelays(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalPulseDelays, HcalPulseDelaysRcd> m_tok;

  HcalPulseDelays* myDBObject;
};

DEFINE_FWK_MODULE(HcalPulseDelaysPopConAnalyzer);
