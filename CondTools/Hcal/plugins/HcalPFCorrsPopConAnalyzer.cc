#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalPFCorrsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalPFCorrsHandler> HcalPFCorrsPopConAnalyzer;

class HcalPFCorrsPopConAnalyzer : public popcon::PopConAnalyzer<HcalPFCorrsHandler> {
public:
  typedef HcalPFCorrsHandler SourceHandler;

  HcalPFCorrsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalPFCorrsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalPFCorrs, HcalPFCorrsRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalPFCorrs(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalPFCorrs, HcalPFCorrsRcd> m_tok;

  HcalPFCorrs* myDBObject;
};

DEFINE_FWK_MODULE(HcalPFCorrsPopConAnalyzer);
