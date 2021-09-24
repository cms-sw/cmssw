#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalChannelStatusHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class ExTestEcalChannelStatusAnalyzer : public popcon::PopConAnalyzer<popcon::EcalChannelStatusHandler> {
public:
  typedef popcon::EcalChannelStatusHandler SourceHandler;

  ExTestEcalChannelStatusAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<popcon::EcalChannelStatusHandler>(pset),
        m_populator(pset),
        m_source(pset.getParameter<edm::ParameterSet>("Source")),
        ecalElectronicsMappingToken_(esConsumes()) {}

private:
  void analyze(const edm::Event& ev, const edm::EventSetup& iSetup) override {
    ecalElectronicsMap = &iSetup.getData(ecalElectronicsMappingToken_);
  }

  void endJob() override {
    m_source.setElectronicsMap(ecalElectronicsMap);
    write();
  }

  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
  const EcalElectronicsMapping* ecalElectronicsMap;
  edm::ESGetToken<EcalElectronicsMapping, EcalMappingRcd> ecalElectronicsMappingToken_;
};

//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalChannelStatusAnalyzer);
