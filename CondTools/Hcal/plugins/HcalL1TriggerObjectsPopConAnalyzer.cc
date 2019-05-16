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
        m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    edm::ESHandle<HcalL1TriggerObjects> objecthandle;
    esetup.get<HcalL1TriggerObjectsRcd>().get(objecthandle);
    myDBObject = new HcalL1TriggerObjects(*objecthandle.product());
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalL1TriggerObjects* myDBObject;
};

DEFINE_FWK_MODULE(HcalL1TriggerObjectsPopConAnalyzer);
