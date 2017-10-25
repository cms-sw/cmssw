#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/CastorGainWidthsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<CastorGainWidthsHandler> CastorGainWidthsPopConAnalyzer;

class CastorGainWidthsPopConAnalyzer: public popcon::PopConAnalyzer<CastorGainWidthsHandler>
{
public:
  typedef CastorGainWidthsHandler SourceHandler;

  CastorGainWidthsPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<CastorGainWidthsHandler>(pset),
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

    edm::ESHandle<CastorGainWidths> objecthandle;
    esetup.get<CastorGainWidthsRcd>().get(objecthandle);
    myDBObject = new CastorGainWidths(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  CastorGainWidths* myDBObject;
};

DEFINE_FWK_MODULE(CastorGainWidthsPopConAnalyzer);
