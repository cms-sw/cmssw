
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"


#include "DQMOffline/Trigger/interface/HLTTagAndProbeEff.h"



#include <vector>
#include <string>


class HLTTagAndProbeOfflineSource : public DQMEDAnalyzer {
 public:
  explicit HLTTagAndProbeOfflineSource(const edm::ParameterSet&);
  ~HLTTagAndProbeOfflineSource()=default;

  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const & run, edm::EventSetup const & c) override;
  virtual void dqmBeginRun(edm::Run const& run, edm::EventSetup const& c) override{}

private:
  std::vector<HLTTagAndProbeEff<reco::GsfElectron,reco::GsfElectronCollection> > tagAndProbeEffs_;

};

HLTTagAndProbeOfflineSource::HLTTagAndProbeOfflineSource(const edm::ParameterSet& config)
{
  auto histCollConfigs =  config.getParameter<std::vector<edm::ParameterSet> >("histCollections");
  for(auto& histCollConfig : histCollConfigs){
    tagAndProbeEffs_.emplace_back(HLTTagAndProbeEff<reco::GsfElectron,reco::GsfElectronCollection>(histCollConfig,consumesCollector()));
  }

}
void HLTTagAndProbeOfflineSource::bookHistograms(DQMStore::IBooker& iBooker,const edm::Run& run,const edm::EventSetup& setup)
{
  for(auto& tpEff : tagAndProbeEffs_) tpEff.bookHists(iBooker);
}

void HLTTagAndProbeOfflineSource::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
  for(auto& tpEff : tagAndProbeEffs_) tpEff.fill(event,setup);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTTagAndProbeOfflineSource);
