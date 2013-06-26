#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalCovarianceMatricesHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalCovarianceMatricesHandler> HcalCovarianceMatricesPopConAnalyzer;

class HcalCovarianceMatricesPopConAnalyzer: public popcon::PopConAnalyzer<HcalCovarianceMatricesHandler>
{
public:
  typedef HcalCovarianceMatricesHandler SourceHandler;

  HcalCovarianceMatricesPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<HcalCovarianceMatricesHandler>(pset),
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

    edm::ESHandle<HcalCovarianceMatrices> objecthandle;
    esetup.get<HcalCovarianceMatricesRcd>().get(objecthandle);
    myDBObject = new HcalCovarianceMatrices(*objecthandle.product() );
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalCovarianceMatrices* myDBObject;
};

DEFINE_FWK_MODULE(HcalCovarianceMatricesPopConAnalyzer);
