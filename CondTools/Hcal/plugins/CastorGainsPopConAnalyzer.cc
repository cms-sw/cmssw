#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/CastorGainsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<CastorGainsHandler> CastorGainsPopConAnalyzer;

class CastorGainsPopConAnalyzer : public popcon::PopConAnalyzer<CastorGainsHandler> {
public:
  typedef CastorGainsHandler SourceHandler;

  CastorGainsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<CastorGainsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<CastorGains, CastorGainsRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new CastorGains(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<CastorGains, CastorGainsRcd> m_tok;
  CastorGains* myDBObject;
};

DEFINE_FWK_MODULE(CastorGainsPopConAnalyzer);
