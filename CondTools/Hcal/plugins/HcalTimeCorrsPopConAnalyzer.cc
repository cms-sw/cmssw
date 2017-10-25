#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalTimeCorrsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalTimeCorrsHandler> HcalTimeCorrsPopConAnalyzer;

class HcalTimeCorrsPopConAnalyzer: public popcon::PopConAnalyzer<HcalTimeCorrsHandler>
{
public:
  typedef HcalTimeCorrsHandler SourceHandler;

  HcalTimeCorrsPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<HcalTimeCorrsHandler>(pset),
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

    edm::ESHandle<HcalTimeCorrs> objecthandle;
    esetup.get<HcalTimeCorrsRcd>().get(objecthandle);
    myDBObject = new HcalTimeCorrs(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalTimeCorrs* myDBObject;
};

DEFINE_FWK_MODULE(HcalTimeCorrsPopConAnalyzer);
