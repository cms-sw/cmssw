#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/CastorRecoParamsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<CastorRecoParamsHandler> CastorRecoParamsPopConAnalyzer;

class CastorRecoParamsPopConAnalyzer : public popcon::PopConAnalyzer<CastorRecoParamsHandler> {
public:
  typedef CastorRecoParamsHandler SourceHandler;

  CastorRecoParamsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<CastorRecoParamsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<CastorRecoParams, CastorRecoParamsRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new CastorRecoParams(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<CastorRecoParams, CastorRecoParamsRcd> m_tok;

  CastorRecoParams* myDBObject;
};

DEFINE_FWK_MODULE(CastorRecoParamsPopConAnalyzer);
