#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/CastorSaturationCorrsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<CastorSaturationCorrsHandler> CastorSaturationCorrsPopConAnalyzer;

class CastorSaturationCorrsPopConAnalyzer: public popcon::PopConAnalyzer<CastorSaturationCorrsHandler>
{
public:
  typedef CastorSaturationCorrsHandler SourceHandler;

  CastorSaturationCorrsPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<CastorSaturationCorrsHandler>(pset),
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

    edm::ESHandle<CastorSaturationCorrs> objecthandle;
    esetup.get<CastorSaturationCorrsRcd>().get(objecthandle);
    myDBObject = new CastorSaturationCorrs(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  CastorSaturationCorrs* myDBObject;
};

DEFINE_FWK_MODULE(CastorSaturationCorrsPopConAnalyzer);
