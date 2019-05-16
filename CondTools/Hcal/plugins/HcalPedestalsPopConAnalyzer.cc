#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalPedestalsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalPedestalsHandler> HcalPedestalsPopConAnalyzer;

class HcalPedestalsPopConAnalyzer : public popcon::PopConAnalyzer<HcalPedestalsHandler> {
public:
  typedef HcalPedestalsHandler SourceHandler;

  HcalPedestalsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalPedestalsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    edm::ESHandle<HcalPedestals> objecthandle;
    esetup.get<HcalPedestalsRcd>().get(objecthandle);
    myDBObject = new HcalPedestals(*objecthandle.product());
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalPedestals* myDBObject;
};

DEFINE_FWK_MODULE(HcalPedestalsPopConAnalyzer);
