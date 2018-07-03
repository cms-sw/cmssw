#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/CastorQIEDataHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<CastorQIEDataHandler> CastorQIEDataPopConAnalyzer;

class CastorQIEDataPopConAnalyzer: public popcon::PopConAnalyzer<CastorQIEDataHandler>
{
public:
  typedef CastorQIEDataHandler SourceHandler;

  CastorQIEDataPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<CastorQIEDataHandler>(pset),
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

    edm::ESHandle<CastorQIEData> objecthandle;
    esetup.get<CastorQIEDataRcd>().get(objecthandle);
    myDBObject = new CastorQIEData(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  CastorQIEData* myDBObject;
};

DEFINE_FWK_MODULE(CastorQIEDataPopConAnalyzer);
