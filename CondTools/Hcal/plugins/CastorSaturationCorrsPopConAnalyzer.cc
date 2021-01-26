#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/CastorSaturationCorrsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<CastorSaturationCorrsHandler> CastorSaturationCorrsPopConAnalyzer;

class CastorSaturationCorrsPopConAnalyzer : public popcon::PopConAnalyzer<CastorSaturationCorrsHandler> {
public:
  typedef CastorSaturationCorrsHandler SourceHandler;

  CastorSaturationCorrsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<CastorSaturationCorrsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<CastorSaturationCorrs, CastorSaturationCorrsRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new CastorSaturationCorrs(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<CastorSaturationCorrs, CastorSaturationCorrsRcd> m_tok;

  CastorSaturationCorrs* myDBObject;
};

DEFINE_FWK_MODULE(CastorSaturationCorrsPopConAnalyzer);
