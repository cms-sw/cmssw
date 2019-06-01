#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalLUTCorrsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalLUTCorrsHandler> HcalLUTCorrsPopConAnalyzer;

class HcalLUTCorrsPopConAnalyzer : public popcon::PopConAnalyzer<HcalLUTCorrsHandler> {
public:
  typedef HcalLUTCorrsHandler SourceHandler;

  HcalLUTCorrsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalLUTCorrsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    edm::ESHandle<HcalLUTCorrs> objecthandle;
    esetup.get<HcalLUTCorrsRcd>().get(objecthandle);
    myDBObject = new HcalLUTCorrs(*objecthandle.product());
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalLUTCorrs* myDBObject;
};

DEFINE_FWK_MODULE(HcalLUTCorrsPopConAnalyzer);
