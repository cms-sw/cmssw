#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalValidationCorrsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalValidationCorrsHandler> HcalValidationCorrsPopConAnalyzer;

class HcalValidationCorrsPopConAnalyzer : public popcon::PopConAnalyzer<HcalValidationCorrsHandler> {
public:
  typedef HcalValidationCorrsHandler SourceHandler;

  HcalValidationCorrsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalValidationCorrsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalValidationCorrs, HcalValidationCorrsRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalValidationCorrs(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalValidationCorrs, HcalValidationCorrsRcd> m_tok;

  HcalValidationCorrs* myDBObject;
};

DEFINE_FWK_MODULE(HcalValidationCorrsPopConAnalyzer);
