#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalQIETypesHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalQIETypesHandler> HcalQIETypesPopConAnalyzer;

class HcalQIETypesPopConAnalyzer : public popcon::PopConAnalyzer<HcalQIETypesHandler> {
public:
  typedef HcalQIETypesHandler SourceHandler;

  HcalQIETypesPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalQIETypesHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalQIETypes, HcalQIETypesRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalQIETypes(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalQIETypes, HcalQIETypesRcd> m_tok;

  HcalQIETypes* myDBObject;
};

DEFINE_FWK_MODULE(HcalQIETypesPopConAnalyzer);
