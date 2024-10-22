#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondFormats/DataRecord/interface/HcalPFCutsRcd.h"
#include "CondTools/Hcal/interface/HcalPFCutsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class HcalPFCutsPopConAnalyzer : public popcon::PopConAnalyzer<HcalPFCutsHandler> {
public:
  typedef HcalPFCutsHandler SourceHandler;

  HcalPFCutsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalPFCutsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalPFCuts, HcalPFCutsRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalPFCuts(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalPFCuts, HcalPFCutsRcd> m_tok;

  HcalPFCuts* myDBObject;
};

DEFINE_FWK_MODULE(HcalPFCutsPopConAnalyzer);
