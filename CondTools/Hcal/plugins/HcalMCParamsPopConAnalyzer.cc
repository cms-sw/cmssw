#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalMCParamsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalMCParamsHandler> HcalMCParamsPopConAnalyzer;

class HcalMCParamsPopConAnalyzer : public popcon::PopConAnalyzer<HcalMCParamsHandler> {
public:
  typedef HcalMCParamsHandler SourceHandler;

  HcalMCParamsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalMCParamsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    edm::ESHandle<HcalMCParams> objecthandle;
    esetup.get<HcalMCParamsRcd>().get(objecthandle);
    myDBObject = new HcalMCParams(*objecthandle.product());
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalMCParams* myDBObject;
};

DEFINE_FWK_MODULE(HcalMCParamsPopConAnalyzer);
