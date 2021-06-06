#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalL1TriggerObjectsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalL1TriggerObjectsHandler> HcalL1TriggerObjectsPopConAnalyzer;

class HcalL1TriggerObjectsPopConAnalyzer : public popcon::PopConAnalyzer<HcalL1TriggerObjectsHandler> {
public:
  typedef HcalL1TriggerObjectsHandler SourceHandler;

  HcalL1TriggerObjectsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalL1TriggerObjectsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalL1TriggerObjects, HcalL1TriggerObjectsRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalL1TriggerObjects(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalL1TriggerObjects, HcalL1TriggerObjectsRcd> m_tok;

  HcalL1TriggerObjects* myDBObject;
};

DEFINE_FWK_MODULE(HcalL1TriggerObjectsPopConAnalyzer);
