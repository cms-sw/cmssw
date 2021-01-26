#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalZDCLowGainFractionsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalZDCLowGainFractionsHandler> HcalZDCLowGainFractionsPopConAnalyzer;

class HcalZDCLowGainFractionsPopConAnalyzer : public popcon::PopConAnalyzer<HcalZDCLowGainFractionsHandler> {
public:
  typedef HcalZDCLowGainFractionsHandler SourceHandler;

  HcalZDCLowGainFractionsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalZDCLowGainFractionsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        m_tok(esConsumes<HcalZDCLowGainFractions, HcalZDCLowGainFractionsRcd>()) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
    delete myDBObject;
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    myDBObject = new HcalZDCLowGainFractions(esetup.getData(m_tok));
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  edm::ESGetToken<HcalZDCLowGainFractions, HcalZDCLowGainFractionsRcd> m_tok;

  HcalZDCLowGainFractions* myDBObject;
};

DEFINE_FWK_MODULE(HcalZDCLowGainFractionsPopConAnalyzer);
