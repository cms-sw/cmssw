#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalCholeskyMatricesHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalCholeskyMatricesHandler> HcalCholeskyMatricesPopConAnalyzer;

class HcalCholeskyMatricesPopConAnalyzer: public popcon::PopConAnalyzer<HcalCholeskyMatricesHandler>
{
public:
  typedef HcalCholeskyMatricesHandler SourceHandler;

  HcalCholeskyMatricesPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<HcalCholeskyMatricesHandler>(pset),
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

    edm::ESHandle<HcalCholeskyMatrices> objecthandle;
    esetup.get<HcalCholeskyMatricesRcd>().get(objecthandle);
    myDBObject = new HcalCholeskyMatrices(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalCholeskyMatrices* myDBObject;
};

DEFINE_FWK_MODULE(HcalCholeskyMatricesPopConAnalyzer);
