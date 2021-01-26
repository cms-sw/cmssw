#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/CastorElectronicsMapHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<CastorElectronicsMapHandler> CastorElectronicsMapPopConAnalyzer;

class CastorElectronicsMapPopConAnalyzer : public popcon::PopConAnalyzer<CastorElectronicsMapHandler> {
public:
  typedef CastorElectronicsMapHandler SourceHandler;

  CastorElectronicsMapPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<CastorElectronicsMapHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<CastorElectronicsMap, CastorElectronicsMapRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new CastorElectronicsMap(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<CastorElectronicsMap, CastorElectronicsMapRcd> m_tok;

  CastorElectronicsMap* myDBObject;
};

DEFINE_FWK_MODULE(CastorElectronicsMapPopConAnalyzer);
