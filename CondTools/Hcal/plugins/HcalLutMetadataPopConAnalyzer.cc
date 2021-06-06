#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalLutMetadataHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalLutMetadataHandler> HcalLutMetadataPopConAnalyzer;

class HcalLutMetadataPopConAnalyzer : public popcon::PopConAnalyzer<HcalLutMetadataHandler> {
public:
  typedef HcalLutMetadataHandler SourceHandler;

  HcalLutMetadataPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalLutMetadataHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalLutMetadata, HcalLutMetadataRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalLutMetadata(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalLutMetadata, HcalLutMetadataRcd> m_tok;

  HcalLutMetadata* myDBObject;
};

DEFINE_FWK_MODULE(HcalLutMetadataPopConAnalyzer);
