#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalTimingParamsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalTimingParamsHandler> HcalTimingParamsPopConAnalyzer;

class HcalTimingParamsPopConAnalyzer: public popcon::PopConAnalyzer<HcalTimingParamsHandler>
{
public:
  typedef HcalTimingParamsHandler SourceHandler;

  HcalTimingParamsPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<HcalTimingParamsHandler>(pset),
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

    edm::ESHandle<HcalTimingParams> objecthandle;
    esetup.get<HcalTimingParamsRcd>().get(objecthandle);
    myDBObject = new HcalTimingParams(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalTimingParams* myDBObject;
};

DEFINE_FWK_MODULE(HcalTimingParamsPopConAnalyzer);
