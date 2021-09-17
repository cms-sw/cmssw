#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalTPParametersHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class HcalTPParametersPopConAnalyzer : public popcon::PopConAnalyzer<HcalTPParametersHandler> {
public:
  typedef HcalTPParametersHandler SourceHandler;

  HcalTPParametersPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalTPParametersHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalTPParameters, HcalTPParametersRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalTPParameters(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalTPParameters, HcalTPParametersRcd> m_tok;

  HcalTPParameters* myDBObject;
};

DEFINE_FWK_MODULE(HcalTPParametersPopConAnalyzer);
