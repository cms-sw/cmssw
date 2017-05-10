#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalQIEDataExtendedHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalQIEDataExtendedHandler> HcalQIEDataExtendedPopConAnalyzer;

class HcalQIEDataExtendedPopConAnalyzer: public popcon::PopConAnalyzer<HcalQIEDataExtendedHandler>
{
public:
  typedef HcalQIEDataExtendedHandler SourceHandler;

  HcalQIEDataExtendedPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<HcalQIEDataExtendedHandler>(pset),
    m_populator(pset),
    m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

private:
  virtual void endJob() override 
  {
    m_source.initObject(myDBObject);
    write();
  }

  virtual void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override
  {
    //Using ES to get the data:

    edm::ESHandle<HcalQIEDataExtended> objecthandle;
    esetup.get<HcalQIEDataExtendedRcd>().get(objecthandle);
    myDBObject = new HcalQIEDataExtended(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalQIEDataExtended* myDBObject;
};

DEFINE_FWK_MODULE(HcalQIEDataExtendedPopConAnalyzer);
