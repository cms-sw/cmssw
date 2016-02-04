#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/CastorChannelQualityHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<CastorChannelQualityHandler> CastorChannelQualityPopConAnalyzer;

class CastorChannelQualityPopConAnalyzer: public popcon::PopConAnalyzer<CastorChannelQualityHandler>
{
public:
  typedef CastorChannelQualityHandler SourceHandler;

  CastorChannelQualityPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<CastorChannelQualityHandler>(pset),
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

    edm::ESHandle<CastorChannelQuality> objecthandle;
    esetup.get<CastorChannelQualityRcd>().get(objecthandle);
    myDBObject = new CastorChannelQuality(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  CastorChannelQuality* myDBObject;
};

DEFINE_FWK_MODULE(CastorChannelQualityPopConAnalyzer);
