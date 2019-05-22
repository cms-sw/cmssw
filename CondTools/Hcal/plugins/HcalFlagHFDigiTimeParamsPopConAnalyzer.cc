#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalFlagHFDigiTimeParamsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef popcon::PopConAnalyzer<HcalFlagHFDigiTimeParamsHandler> HcalFlagHFDigiTimeParamsPopConAnalyzer;

class HcalFlagHFDigiTimeParamsPopConAnalyzer : public popcon::PopConAnalyzer<HcalFlagHFDigiTimeParamsHandler> {
public:
  typedef HcalFlagHFDigiTimeParamsHandler SourceHandler;

  HcalFlagHFDigiTimeParamsPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<HcalFlagHFDigiTimeParamsHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

private:
  void endJob() override {
    m_source.initObject(myDBObject);
    write();
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    //Using ES to get the data:

    edm::ESHandle<HcalFlagHFDigiTimeParams> objecthandle;
    esetup.get<HcalFlagHFDigiTimeParamsRcd>().get(objecthandle);
    myDBObject = new HcalFlagHFDigiTimeParams(*objecthandle.product());
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  HcalFlagHFDigiTimeParams* myDBObject;
};

DEFINE_FWK_MODULE(HcalFlagHFDigiTimeParamsPopConAnalyzer);
