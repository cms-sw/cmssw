
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "TrigObjTnPHistColl.h"

class TrigObjTnPSource : public DQMEDAnalyzer {
public:
  explicit TrigObjTnPSource(const edm::ParameterSet&);
  ~TrigObjTnPSource() override = default;
  TrigObjTnPSource(const TrigObjTnPSource&) = delete; 
  TrigObjTnPSource& operator=(const TrigObjTnPSource&) = delete; 
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const & run, edm::EventSetup const & c) override;
  void dqmBeginRun(edm::Run const& run, edm::EventSetup const& c) override{}

private:
  edm::EDGetTokenT<trigger::TriggerEvent> trigEvtToken_;
  std::vector<TrigObjTnPHistColl> tnpHistColls_;
  
};

TrigObjTnPSource::TrigObjTnPSource(const edm::ParameterSet& config):
  trigEvtToken_(consumes<trigger::TriggerEvent>(config.getParameter<edm::InputTag>("triggerEvent")))
{
  auto histCollConfigs = config.getParameter<std::vector<edm::ParameterSet>>("histColls");
  for(auto& histCollConfig : histCollConfigs) tnpHistColls_.emplace_back(TrigObjTnPHistColl(histCollConfig,consumesCollector()));
    
}


void TrigObjTnPSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("triggerEvent", edm::InputTag("hltTriggerSummaryAOD"));
  desc.addVPSet("histColls",
		TrigObjTnPHistColl::makePSetDescription(),
		std::vector<edm::ParameterSet>());  
  descriptions.add("trigObjTnPSource",desc);
}

void TrigObjTnPSource::bookHistograms(DQMStore::IBooker& iBooker,const edm::Run& run,
						 const edm::EventSetup& setup)
{
  for(auto& histColl : tnpHistColls_) histColl.bookHists(iBooker);
}


void TrigObjTnPSource::analyze(const edm::Event& event,const edm::EventSetup& setup)
{
  edm::Handle<trigger::TriggerEvent> trigEvtHandle;
  event.getByToken(trigEvtToken_,trigEvtHandle);
  for(auto& histColl: tnpHistColls_) histColl.fill(*trigEvtHandle,event,setup);
}

DEFINE_FWK_MODULE(TrigObjTnPSource);
