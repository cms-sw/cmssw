
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "TrigObjTnPHistColl.h"

class TrigObjTnPSource : public DQMGlobalEDAnalyzer<std::vector<TrigObjTnPHistColl>> {
public:
  explicit TrigObjTnPSource(const edm::ParameterSet&);
  ~TrigObjTnPSource() override = default;
  TrigObjTnPSource(const TrigObjTnPSource&) = delete;
  TrigObjTnPSource& operator=(const TrigObjTnPSource&) = delete;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void dqmAnalyze(const edm::Event&, const edm::EventSetup&, const std::vector<TrigObjTnPHistColl>&) const override;
  void bookHistograms(DQMStore::IBooker&,
                      const edm::Run&,
                      const edm::EventSetup&,
                      std::vector<TrigObjTnPHistColl>&) const override;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&, std::vector<TrigObjTnPHistColl>&) const override;

private:
  edm::EDGetTokenT<trigger::TriggerEvent> trigEvtToken_;
  edm::EDGetTokenT<edm::TriggerResults> trigResultsToken_;
  std::string hltProcess_;
  //it would be more memory efficient to save this as a vector of TrigTnPHistColls and then use this to instance
  //the TrigTnPHistColls for each run, something to do for the future
  std::vector<edm::ParameterSet> histCollConfigs_;
};

TrigObjTnPSource::TrigObjTnPSource(const edm::ParameterSet& config)
    : trigEvtToken_(consumes<trigger::TriggerEvent>(config.getParameter<edm::InputTag>("triggerEvent"))),
      trigResultsToken_(consumes<edm::TriggerResults>(config.getParameter<edm::InputTag>("triggerResults"))),
      hltProcess_(config.getParameter<edm::InputTag>("triggerResults").process()),
      histCollConfigs_(config.getParameter<std::vector<edm::ParameterSet>>("histColls")) {}

void TrigObjTnPSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("triggerEvent", edm::InputTag("hltTriggerSummaryAOD::HLT"));
  desc.add<edm::InputTag>("triggerResults", edm::InputTag("TriggerResults::HLT"));
  desc.addVPSet("histColls", TrigObjTnPHistColl::makePSetDescription(), std::vector<edm::ParameterSet>());
  descriptions.add("trigObjTnPSource", desc);
}

void TrigObjTnPSource::bookHistograms(DQMStore::IBooker& iBooker,
                                      const edm::Run& run,
                                      const edm::EventSetup& setup,
                                      std::vector<TrigObjTnPHistColl>& tnpHistColls) const {
  tnpHistColls.clear();
  HLTConfigProvider hltConfig;
  bool hltChanged = false;
  hltConfig.init(run, setup, hltProcess_, hltChanged);
  for (auto& histCollConfig : histCollConfigs_)
    tnpHistColls.emplace_back(TrigObjTnPHistColl(histCollConfig));
  for (auto& histColl : tnpHistColls) {
    histColl.init(hltConfig);
    histColl.bookHists(iBooker);
  }
}

void TrigObjTnPSource::dqmBeginRun(const edm::Run& run,
                                   const edm::EventSetup& setup,
                                   std::vector<TrigObjTnPHistColl>& tnpHistColls) const {}

void TrigObjTnPSource::dqmAnalyze(const edm::Event& event,
                                  const edm::EventSetup& setup,
                                  const std::vector<TrigObjTnPHistColl>& tnpHistColls) const {
  edm::Handle<trigger::TriggerEvent> trigEvtHandle;
  event.getByToken(trigEvtToken_, trigEvtHandle);
  edm::Handle<edm::TriggerResults> trigResultsHandle;
  event.getByToken(trigResultsToken_, trigResultsHandle);
  //DQM should never crash the HLT under any circumstances (except at configure)
  if (trigEvtHandle.isValid() && trigResultsHandle.isValid()) {
    for (auto& histColl : tnpHistColls) {
      histColl.fill(*trigEvtHandle, *trigResultsHandle, event.triggerNames(*trigResultsHandle));
    }
  }
}

DEFINE_FWK_MODULE(TrigObjTnPSource);
