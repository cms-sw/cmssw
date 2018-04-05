#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/CastorElectronicsMapHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<CastorElectronicsMapHandler> CastorElectronicsMapPopConAnalyzer;

class CastorElectronicsMapPopConAnalyzer: public popcon::PopConAnalyzer<CastorElectronicsMapHandler>
{
public:
  typedef CastorElectronicsMapHandler SourceHandler;

  CastorElectronicsMapPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<CastorElectronicsMapHandler>(pset),
    m_populator(pset),
    m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

private:
  void endJob() override 
  {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override
  {
    //Using ES to get the data:

    edm::ESHandle<CastorElectronicsMap> objecthandle;
    esetup.get<CastorElectronicsMapRcd>().get(objecthandle);
    myDBObject = new CastorElectronicsMap(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  CastorElectronicsMap* myDBObject;
};

DEFINE_FWK_MODULE(CastorElectronicsMapPopConAnalyzer);
