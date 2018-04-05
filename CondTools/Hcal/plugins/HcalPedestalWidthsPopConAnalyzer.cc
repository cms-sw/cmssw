#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalPedestalWidthsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalPedestalWidthsHandler> HcalPedestalWidthsPopConAnalyzer;

class HcalPedestalWidthsPopConAnalyzer: public popcon::PopConAnalyzer<HcalPedestalWidthsHandler>
{
public:
  typedef HcalPedestalWidthsHandler SourceHandler;

  HcalPedestalWidthsPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<HcalPedestalWidthsHandler>(pset),
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

    edm::ESHandle<HcalPedestalWidths> objecthandle;
    esetup.get<HcalPedestalWidthsRcd>().get(objecthandle);
    myDBObject = new HcalPedestalWidths(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalPedestalWidths* myDBObject;
};

DEFINE_FWK_MODULE(HcalPedestalWidthsPopConAnalyzer);
