
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

#include "DQMOffline/Trigger/interface/HLTDQMTagAndProbeEff.h"



#include <vector>
#include <string>


template <typename ObjType,typename ObjCollType> 
class HLTTagAndProbeOfflineSource : public DQMEDAnalyzer {
 public:
  explicit HLTTagAndProbeOfflineSource(const edm::ParameterSet&);
  ~HLTTagAndProbeOfflineSource()=default;
  HLTTagAndProbeOfflineSource(const HLTTagAndProbeOfflineSource&)=delete; 
  HLTTagAndProbeOfflineSource& operator=(const HLTTagAndProbeOfflineSource&)=delete; 
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const & run, edm::EventSetup const & c) override;
  virtual void dqmBeginRun(edm::Run const& run, edm::EventSetup const& c) override{}

private:
  std::vector<HLTDQMTagAndProbeEff<ObjType,ObjCollType> > tagAndProbeEffs_;

};

template <typename ObjType,typename ObjCollType> 
HLTTagAndProbeOfflineSource<ObjType,ObjCollType>::
HLTTagAndProbeOfflineSource(const edm::ParameterSet& config)
{
  auto histCollConfigs =  config.getParameter<std::vector<edm::ParameterSet> >("tagAndProbeCollections");
  for(auto& histCollConfig : histCollConfigs){
    tagAndProbeEffs_.emplace_back(HLTDQMTagAndProbeEff<ObjType,ObjCollType>(histCollConfig,consumesCollector()));
  }
}


template <typename ObjType,typename ObjCollType> 
void HLTTagAndProbeOfflineSource<ObjType,ObjCollType>::
fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("objs", edm::InputTag(""));
  desc.addVPSet("tagAndProbeCollections",
		HLTDQMTagAndProbeEff<ObjType,ObjCollType>::makePSetDescription(),
		std::vector<edm::ParameterSet>());
  descriptions.add("hltTagAndProbeOfflineSource", desc);
  
}

template <typename ObjType,typename ObjCollType> 
void HLTTagAndProbeOfflineSource<ObjType,ObjCollType>::
bookHistograms(DQMStore::IBooker& iBooker,const edm::Run& run,const edm::EventSetup& setup)
{
  for(auto& tpEff : tagAndProbeEffs_) tpEff.bookHists(iBooker);
}

template <typename ObjType,typename ObjCollType> 
void HLTTagAndProbeOfflineSource<ObjType,ObjCollType>::
analyze(const edm::Event& event,const edm::EventSetup& setup)
{
  for(auto& tpEff : tagAndProbeEffs_) tpEff.fill(event,setup);
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
using HLTEleTagAndProbeOfflineSource = HLTTagAndProbeOfflineSource<reco::GsfElectron,reco::GsfElectronCollection>;
DEFINE_FWK_MODULE(HLTEleTagAndProbeOfflineSource);
