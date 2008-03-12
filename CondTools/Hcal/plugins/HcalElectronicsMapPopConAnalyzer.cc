#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalElectronicsMapHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalElectronicsMapHandler> HcalElectronicsMapPopConAnalyzer;

class HcalElectronicsMapPopConAnalyzer: public popcon::PopConAnalyzer<HcalElectronicsMapHandler>
{
public:
  typedef HcalElectronicsMapHandler SourceHandler;

  HcalElectronicsMapPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<HcalElectronicsMapHandler>(pset),
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

    edm::ESHandle<HcalElectronicsMap> objecthandle;
    esetup.get<HcalElectronicsMapRcd>().get(objecthandle);
    myDBObject = new HcalElectronicsMap(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalElectronicsMap* myDBObject;
};

DEFINE_FWK_MODULE(HcalElectronicsMapPopConAnalyzer);
