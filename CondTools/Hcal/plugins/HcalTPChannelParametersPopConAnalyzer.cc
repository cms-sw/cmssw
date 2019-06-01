#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalTPChannelParametersHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class HcalTPChannelParametersPopConAnalyzer : public popcon::PopConAnalyzer<HcalTPChannelParametersHandler> {
public:
  typedef HcalTPChannelParametersHandler SourceHandler;

  HcalTPChannelParametersPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalTPChannelParametersHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    edm::ESHandle<HcalTPChannelParameters> objecthandle;
    esetup.get<HcalTPChannelParametersRcd>().get(objecthandle);
    myDBObject = new HcalTPChannelParameters(*objecthandle.product());
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalTPChannelParameters* myDBObject;
};

DEFINE_FWK_MODULE(HcalTPChannelParametersPopConAnalyzer);
