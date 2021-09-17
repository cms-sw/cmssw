#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/CastorChannelQualityHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<CastorChannelQualityHandler> CastorChannelQualityPopConAnalyzer;

class CastorChannelQualityPopConAnalyzer : public popcon::PopConAnalyzer<CastorChannelQualityHandler> {
public:
  typedef CastorChannelQualityHandler SourceHandler;

  CastorChannelQualityPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<CastorChannelQualityHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<CastorChannelQuality, CastorChannelQualityRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new CastorChannelQuality(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<CastorChannelQuality, CastorChannelQualityRcd> m_tok;

  CastorChannelQuality* myDBObject;
};

DEFINE_FWK_MODULE(CastorChannelQualityPopConAnalyzer);
