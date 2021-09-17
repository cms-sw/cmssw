#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/CastorPedestalsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<CastorPedestalsHandler> CastorPedestalsPopConAnalyzer;

class CastorPedestalsPopConAnalyzer : public popcon::PopConAnalyzer<CastorPedestalsHandler> {
public:
  typedef CastorPedestalsHandler SourceHandler;

  CastorPedestalsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<CastorPedestalsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<CastorPedestals, CastorPedestalsRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new CastorPedestals(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<CastorPedestals, CastorPedestalsRcd> m_tok;

  CastorPedestals* myDBObject;
};

DEFINE_FWK_MODULE(CastorPedestalsPopConAnalyzer);
