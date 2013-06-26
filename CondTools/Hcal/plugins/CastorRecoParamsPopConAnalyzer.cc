#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/CastorRecoParamsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<CastorRecoParamsHandler> CastorRecoParamsPopConAnalyzer;

class CastorRecoParamsPopConAnalyzer: public popcon::PopConAnalyzer<CastorRecoParamsHandler>
{
public:
  typedef CastorRecoParamsHandler SourceHandler;

  CastorRecoParamsPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<CastorRecoParamsHandler>(pset),
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

    edm::ESHandle<CastorRecoParams> objecthandle;
    esetup.get<CastorRecoParamsRcd>().get(objecthandle);
    myDBObject = new CastorRecoParams(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  CastorRecoParams* myDBObject;
};

DEFINE_FWK_MODULE(CastorRecoParamsPopConAnalyzer);
