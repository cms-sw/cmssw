#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalTimingParamsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalTimingParamsHandler> HcalTimingParamsPopConAnalyzer;

class HcalTimingParamsPopConAnalyzer : public popcon::PopConAnalyzer<HcalTimingParamsHandler> {
public:
  typedef HcalTimingParamsHandler SourceHandler;

  HcalTimingParamsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalTimingParamsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalTimingParams, HcalTimingParamsRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalTimingParams(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalTimingParams, HcalTimingParamsRcd> m_tok;

  HcalTimingParams* myDBObject;
};

DEFINE_FWK_MODULE(HcalTimingParamsPopConAnalyzer);
