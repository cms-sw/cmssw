#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalDcsValuesHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalDcsValuesHandler> HcalDcsValuesPopConAnalyzer;

class HcalDcsValuesPopConAnalyzer: public popcon::PopConAnalyzer<HcalDcsValuesHandler>
{
public:
  typedef HcalDcsValuesHandler SourceHandler;

  HcalDcsValuesPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<HcalDcsValuesHandler>(pset),
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

    edm::ESHandle<HcalDcsValues> objecthandle;
    esetup.get<HcalDcsRcd>().get(objecthandle);
    myDBObject = new HcalDcsValues(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalDcsValues* myDBObject;
};

DEFINE_FWK_MODULE(HcalDcsValuesPopConAnalyzer);
