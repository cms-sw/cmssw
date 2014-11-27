#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalODFCorrectionsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"


class HcalODFCorrectionsPopConAnalyzer: public popcon::PopConAnalyzer<HcalODFCorrectionsHandler>
{
public:
  typedef HcalODFCorrectionsHandler SourceHandler;

  HcalODFCorrectionsPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<HcalODFCorrectionsHandler>(pset),
    m_populator(pset),
    m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

private:
  virtual void endJob() override 
  {
    m_source.initObject(myDBObject);
    write();
    delete myDBObject;
  }

  virtual void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override
  {
    //Using ES to get the data:

    edm::ESHandle<HcalODFCorrections> objecthandle;
    esetup.get<HcalODFCorrectionsRcd>().get(objecthandle);
    myDBObject = new HcalODFCorrections(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalODFCorrections* myDBObject;
};

DEFINE_FWK_MODULE(HcalODFCorrectionsPopConAnalyzer);
