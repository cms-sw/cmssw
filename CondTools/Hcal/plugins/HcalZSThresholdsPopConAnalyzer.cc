#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalZSThresholdsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalZSThresholdsHandler> HcalZSThresholdsPopConAnalyzer;

class HcalZSThresholdsPopConAnalyzer : public popcon::PopConAnalyzer<HcalZSThresholdsHandler> {
public:
  typedef HcalZSThresholdsHandler SourceHandler;

  HcalZSThresholdsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalZSThresholdsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    edm::ESHandle<HcalZSThresholds> objecthandle;
    esetup.get<HcalZSThresholdsRcd>().get(objecthandle);
    myDBObject = new HcalZSThresholds(*objecthandle.product());
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalZSThresholds* myDBObject;
};

DEFINE_FWK_MODULE(HcalZSThresholdsPopConAnalyzer);
