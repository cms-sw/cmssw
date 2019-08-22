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
        m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    edm::ESHandle<HcalRespCorrs> objecthandle;
    esetup.get<HcalRespCorrsRcd>().get(objecthandle);
    myDBObject = new HcalRespCorrs(*objecthandle.product());
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalRespCorrs* myDBObject;
};

DEFINE_FWK_MODULE(HcalRespCorrsPopConAnalyzer);
