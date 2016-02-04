#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalGainWidthsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalGainWidthsHandler> HcalGainWidthsPopConAnalyzer;

class HcalGainWidthsPopConAnalyzer: public popcon::PopConAnalyzer<HcalGainWidthsHandler>
{
public:
  typedef HcalGainWidthsHandler SourceHandler;

  HcalGainWidthsPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<HcalGainWidthsHandler>(pset),
    m_populator(pset),
    m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

private:
  virtual void endJob() 
  {
    m_source.initObject(myDBObject);
    write();
  }

  virtual void analyze(const edm::Event& ev, const edm::EventSetup& esetup)
  {
    //Using ES to get the data:

    edm::ESHandle<HcalGainWidths> objecthandle;
    esetup.get<HcalGainWidthsRcd>().get(objecthandle);
    myDBObject = new HcalGainWidths(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalGainWidths* myDBObject;
};

DEFINE_FWK_MODULE(HcalGainWidthsPopConAnalyzer);
