#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalLongRecoParamsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalLongRecoParamsHandler> HcalLongRecoParamsPopConAnalyzer;

class HcalLongRecoParamsPopConAnalyzer: public popcon::PopConAnalyzer<HcalLongRecoParamsHandler>
{
public:
  typedef HcalLongRecoParamsHandler SourceHandler;

  HcalLongRecoParamsPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<HcalLongRecoParamsHandler>(pset),
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

    edm::ESHandle<HcalLongRecoParams> objecthandle;
    esetup.get<HcalLongRecoParamsRcd>().get(objecthandle);
    myDBObject = new HcalLongRecoParams(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalLongRecoParams* myDBObject;
};

DEFINE_FWK_MODULE(HcalLongRecoParamsPopConAnalyzer);
