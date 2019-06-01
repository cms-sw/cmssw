#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalSiPMParametersHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class HcalSiPMParametersPopConAnalyzer : public popcon::PopConAnalyzer<HcalSiPMParametersHandler> {
public:
  typedef HcalSiPMParametersHandler SourceHandler;

  HcalSiPMParametersPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalSiPMParametersHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    edm::ESHandle<HcalSiPMParameters> objecthandle;
    esetup.get<HcalSiPMParametersRcd>().get(objecthandle);
    myDBObject = new HcalSiPMParameters(*objecthandle.product());
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalSiPMParameters* myDBObject;
};

DEFINE_FWK_MODULE(HcalSiPMParametersPopConAnalyzer);
