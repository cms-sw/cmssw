#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/CastorPedestalWidthsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<CastorPedestalWidthsHandler> CastorPedestalWidthsPopConAnalyzer;

class CastorPedestalWidthsPopConAnalyzer: public popcon::PopConAnalyzer<CastorPedestalWidthsHandler>
{
public:
  typedef CastorPedestalWidthsHandler SourceHandler;

  CastorPedestalWidthsPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<CastorPedestalWidthsHandler>(pset),
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

    edm::ESHandle<CastorPedestalWidths> objecthandle;
    esetup.get<CastorPedestalWidthsRcd>().get(objecthandle);
    myDBObject = new CastorPedestalWidths(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  CastorPedestalWidths* myDBObject;
};

DEFINE_FWK_MODULE(CastorPedestalWidthsPopConAnalyzer);
