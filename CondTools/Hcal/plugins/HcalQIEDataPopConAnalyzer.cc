#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalQIEDataHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalQIEDataHandler> HcalQIEDataPopConAnalyzer;

class HcalQIEDataPopConAnalyzer: public popcon::PopConAnalyzer<HcalQIEDataHandler>
{
public:
  typedef HcalQIEDataHandler SourceHandler;

  HcalQIEDataPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<HcalQIEDataHandler>(pset),
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

    edm::ESHandle<HcalQIEData> objecthandle;
    esetup.get<HcalQIEDataRcd>().get(objecthandle);
    myDBObject = new HcalQIEData(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalQIEData* myDBObject;
};

DEFINE_FWK_MODULE(HcalQIEDataPopConAnalyzer);
