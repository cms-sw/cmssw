#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalElectronicsMapHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalElectronicsMapHandler> HcalElectronicsMapPopConAnalyzer;

class HcalElectronicsMapPopConAnalyzer : public popcon::PopConAnalyzer<HcalElectronicsMapHandler> {
public:
  typedef HcalElectronicsMapHandler SourceHandler;

  HcalElectronicsMapPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalElectronicsMapHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalElectronicsMap, HcalElectronicsMapRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalElectronicsMap(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalElectronicsMap, HcalElectronicsMapRcd> m_tok;

  HcalElectronicsMap* myDBObject;
};

DEFINE_FWK_MODULE(HcalElectronicsMapPopConAnalyzer);
