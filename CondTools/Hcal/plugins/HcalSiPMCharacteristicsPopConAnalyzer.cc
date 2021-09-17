#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalSiPMCharacteristicsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class HcalSiPMCharacteristicsPopConAnalyzer : public popcon::PopConAnalyzer<HcalSiPMCharacteristicsHandler> {
public:
  typedef HcalSiPMCharacteristicsHandler SourceHandler;

  HcalSiPMCharacteristicsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalSiPMCharacteristicsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalSiPMCharacteristics, HcalSiPMCharacteristicsRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalSiPMCharacteristics(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalSiPMCharacteristics, HcalSiPMCharacteristicsRcd> m_tok;

  HcalSiPMCharacteristics* myDBObject;
};

DEFINE_FWK_MODULE(HcalSiPMCharacteristicsPopConAnalyzer);
