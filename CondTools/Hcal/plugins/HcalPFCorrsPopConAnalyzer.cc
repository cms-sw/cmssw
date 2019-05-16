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
        m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    edm::ESHandle<HcalPFCorrs> objecthandle;
    esetup.get<HcalPFCorrsRcd>().get(objecthandle);
    myDBObject = new HcalPFCorrs(*objecthandle.product());
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalPFCorrs* myDBObject;
};

DEFINE_FWK_MODULE(HcalPFCorrsPopConAnalyzer);
