#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalRecoParamsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalRecoParamsHandler> HcalRecoParamsPopConAnalyzer;

class HcalRecoParamsPopConAnalyzer: public popcon::PopConAnalyzer<HcalRecoParamsHandler>
{
public:
  typedef HcalRecoParamsHandler SourceHandler;

  HcalRecoParamsPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<HcalRecoParamsHandler>(pset),
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

    edm::ESHandle<HcalRecoParams> objecthandle;
    esetup.get<HcalRecoParamsRcd>().get(objecthandle);
    myDBObject = new HcalRecoParams(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalRecoParams* myDBObject;
};

DEFINE_FWK_MODULE(HcalRecoParamsPopConAnalyzer);
