#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalZDCLowGainFractionsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalZDCLowGainFractionsHandler> HcalZDCLowGainFractionsPopConAnalyzer;

class HcalZDCLowGainFractionsPopConAnalyzer: public popcon::PopConAnalyzer<HcalZDCLowGainFractionsHandler>
{
public:
  typedef HcalZDCLowGainFractionsHandler SourceHandler;

  HcalZDCLowGainFractionsPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<HcalZDCLowGainFractionsHandler>(pset),
    m_populator(pset),
    m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

private:
  virtual void endJob() override 
  {
    m_source.initObject(myDBObject);
    write();
    delete myDBObject;
  }

  virtual void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override
  {
    //Using ES to get the data:

    edm::ESHandle<HcalZDCLowGainFractions> objecthandle;
    esetup.get<HcalZDCLowGainFractionsRcd>().get(objecthandle);
    
    myDBObject = new HcalZDCLowGainFractions(*objecthandle.product());
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalZDCLowGainFractions* myDBObject;
};

DEFINE_FWK_MODULE(HcalZDCLowGainFractionsPopConAnalyzer);
