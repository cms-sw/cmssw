#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalRespCorrsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalRespCorrsHandler> HcalRespCorrsPopConAnalyzer;

class HcalRespCorrsPopConAnalyzer : public popcon::PopConAnalyzer<HcalRespCorrsHandler> {
public:
  typedef HcalRespCorrsHandler SourceHandler;

  HcalRespCorrsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalRespCorrsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalRespCorrs, HcalRespCorrsRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalRespCorrs(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalRespCorrs, HcalRespCorrsRcd> m_tok;

  HcalRespCorrs* myDBObject;
};

DEFINE_FWK_MODULE(HcalRespCorrsPopConAnalyzer);
