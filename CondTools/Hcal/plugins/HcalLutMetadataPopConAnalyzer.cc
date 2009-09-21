#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalLutMetadataHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalLutMetadataHandler> HcalLutMetadataPopConAnalyzer;

class HcalLutMetadataPopConAnalyzer: public popcon::PopConAnalyzer<HcalLutMetadataHandler>
{
public:
  typedef HcalLutMetadataHandler SourceHandler;

  HcalLutMetadataPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<HcalLutMetadataHandler>(pset),
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

    edm::ESHandle<HcalLutMetadata> objecthandle;
    esetup.get<HcalLutMetadataRcd>().get(objecthandle);
    myDBObject = new HcalLutMetadata(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalLutMetadata* myDBObject;
};

DEFINE_FWK_MODULE(HcalLutMetadataPopConAnalyzer);
