/*
  TagAndProbeBtagTriggerMonitor DQM code
*/
//
// Originally created by:  Roberval Walsh
//                         June 2017
#include <memory>

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/transform.h"

class TagAndProbeBtagTriggerMonitor : public DQMEDAnalyzer {
public:
  TagAndProbeBtagTriggerMonitor(const edm::ParameterSet&);
  ~TagAndProbeBtagTriggerMonitor() throw() override;

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

  struct JetRefCompare {
    inline bool operator()(const edm::RefToBase<reco::Jet>& j1, const edm::RefToBase<reco::Jet>& j2) const {
      return (j1.id() < j2.id()) || ((j1.id() == j2.id()) && (j1.key() < j2.key()));
    }
  };
  typedef std::map<edm::RefToBase<reco::Jet>, float, JetRefCompare> JetTagMap;

private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  std::string processname_;
  std::string triggerobjbtag_;

  double jetPtmin_;
  double jetEtamax_;
  double tagBtagmin_;
  double probeBtagmin_;

  std::vector<double> jetPtbins_;
  std::vector<double> jetEtabins_;
  std::vector<double> jetPhibins_;
  std::vector<double> jetBtagbins_;

  edm::InputTag triggerSummaryLabel_;

  std::vector<edm::EDGetTokenT<reco::JetTagCollection> > jetTagTokens_;

  edm::EDGetTokenT<trigger::TriggerEvent> triggerSummaryToken_;

  MonitorElement* pt_jet1_;
  MonitorElement* pt_jet2_;
  MonitorElement* eta_jet1_;
  MonitorElement* eta_jet2_;
  MonitorElement* phi_jet1_;
  MonitorElement* phi_jet2_;
  MonitorElement* eta_phi_jet1_;
  MonitorElement* eta_phi_jet2_;

  MonitorElement* pt_probe_;
  MonitorElement* pt_probe_match_;
  MonitorElement* eta_probe_;
  MonitorElement* eta_probe_match_;
  MonitorElement* phi_probe_;
  MonitorElement* phi_probe_match_;
  MonitorElement* eta_phi_probe_;
  MonitorElement* eta_phi_probe_match_;

  MonitorElement* discr_offline_btag_jet1_;
  MonitorElement* discr_offline_btag_jet2_;

  std::unique_ptr<GenericTriggerEventFlag> genTriggerEventFlag_;  // tag & probe: trigger flag for num and den
};

TagAndProbeBtagTriggerMonitor::TagAndProbeBtagTriggerMonitor(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("dirname")),
      requireValidHLTPaths_(iConfig.getParameter<bool>("requireValidHLTPaths")),
      hltPathsAreValid_(false) {
  processname_ = iConfig.getParameter<std::string>("processname");
  triggerobjbtag_ = iConfig.getParameter<std::string>("triggerobjbtag");
  jetPtmin_ = iConfig.getParameter<double>("jetPtMin");
  jetEtamax_ = iConfig.getParameter<double>("jetEtaMax");
  tagBtagmin_ = iConfig.getParameter<double>("tagBtagMin");
  probeBtagmin_ = iConfig.getParameter<double>("probeBtagMin");
  triggerSummaryLabel_ = iConfig.getParameter<edm::InputTag>("triggerSummary");
  triggerSummaryToken_ = consumes<trigger::TriggerEvent>(triggerSummaryLabel_);
  // New
  jetTagTokens_ =
      edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag> >("btagAlgos"),
                            [this](edm::InputTag const& tag) { return mayConsume<reco::JetTagCollection>(tag); });

  genTriggerEventFlag_ = std::make_unique<GenericTriggerEventFlag>(
      iConfig.getParameter<edm::ParameterSet>("genericTriggerEventPSet"), consumesCollector(), *this);

  jetPtbins_ = iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetPt");
  jetEtabins_ = iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetEta");
  jetPhibins_ = iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetPhi");
  jetBtagbins_ = iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetBtag");
}

TagAndProbeBtagTriggerMonitor::~TagAndProbeBtagTriggerMonitor() throw() {
  if (genTriggerEventFlag_) {
    genTriggerEventFlag_.reset();
  }
}

void TagAndProbeBtagTriggerMonitor::bookHistograms(DQMStore::IBooker& ibooker,
                                                   edm::Run const& iRun,
                                                   edm::EventSetup const& iSetup) {
  // Initialize the GenericTriggerEventFlag
  if (genTriggerEventFlag_ && genTriggerEventFlag_->on())
    genTriggerEventFlag_->initRun(iRun, iSetup);

  // check if every HLT path specified in numerator and denominator has a valid match in the HLT Menu
  hltPathsAreValid_ =
      (genTriggerEventFlag_ && genTriggerEventFlag_->on() && genTriggerEventFlag_->allHLTPathsAreValid());

  // if valid HLT paths are required,
  // create DQM outputs only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  std::string currentFolder = folderName_;
  ibooker.setCurrentFolder(currentFolder);

  int ptnbins = jetPtbins_.size() - 1;
  int etanbins = jetEtabins_.size() - 1;
  int phinbins = jetPhibins_.size() - 1;
  int btagnbins = jetBtagbins_.size() - 1;

  std::vector<float> fptbins(jetPtbins_.begin(), jetPtbins_.end());
  std::vector<float> fetabins(jetEtabins_.begin(), jetEtabins_.end());
  std::vector<float> fphibins(jetPhibins_.begin(), jetPhibins_.end());
  std::vector<float> fbtagbins(jetBtagbins_.begin(), jetBtagbins_.end());

  float* ptbins = &fptbins[0];
  float* etabins = &fetabins[0];
  float* phibins = &fphibins[0];
  float* btagbins = &fbtagbins[0];

  pt_jet1_ = ibooker.book1D("pt_jet1", "pt_jet1", ptnbins, ptbins);
  pt_jet2_ = ibooker.book1D("pt_jet2", "pt_jet2", ptnbins, ptbins);
  eta_jet1_ = ibooker.book1D("eta_jet1", "eta_jet1", etanbins, etabins);
  eta_jet2_ = ibooker.book1D("eta_jet2", "eta_jet2", etanbins, etabins);
  phi_jet1_ = ibooker.book1D("phi_jet1", "phi_jet1", phinbins, phibins);
  phi_jet2_ = ibooker.book1D("phi_jet2", "phi_jet2", phinbins, phibins);
  eta_phi_jet1_ = ibooker.book2D("eta_phi_jet1", "eta_phi_jet1", etanbins, etabins, phinbins, phibins);
  eta_phi_jet2_ = ibooker.book2D("eta_phi_jet2", "eta_phi_jet2", etanbins, etabins, phinbins, phibins);

  pt_probe_ = ibooker.book1D("pt_probe", "pt_probe", ptnbins, ptbins);
  pt_probe_match_ = ibooker.book1D("pt_probe_match", "pt_probe_match", ptnbins, ptbins);
  eta_probe_ = ibooker.book1D("eta_probe", "eta_probe", etanbins, etabins);
  eta_probe_match_ = ibooker.book1D("eta_probe_match", "eta_probe_match", etanbins, etabins);
  phi_probe_ = ibooker.book1D("phi_probe", "phi_probe", phinbins, phibins);
  phi_probe_match_ = ibooker.book1D("phi_probe_match", "phi_probe_match", phinbins, phibins);
  eta_phi_probe_ = ibooker.book2D("eta_phi_probe", "eta_phi_probe", etanbins, etabins, phinbins, phibins);
  eta_phi_probe_match_ = ibooker.book2D("eta_phi_probe_match", "eta_phi_match", etanbins, etabins, phinbins, phibins);

  discr_offline_btag_jet1_ = ibooker.book1D("discr_offline_btag_jet1", "discr_offline_btag_jet1", btagnbins, btagbins);
  discr_offline_btag_jet2_ = ibooker.book1D("discr_offline_btag_jet2", "discr_offline_btag_jet2", btagnbins, btagbins);
}

void TagAndProbeBtagTriggerMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  // if valid HLT paths are required,
  // analyze event only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  bool match1 = false;
  bool match2 = false;

  JetTagMap allJetBTagVals;
  for (const auto& jetTagToken : jetTagTokens_) {
    edm::Handle<reco::JetTagCollection> bjetHandle;
    iEvent.getByToken(jetTagToken, bjetHandle);
    if (not bjetHandle.isValid()) {
      edm::LogWarning("TagAndProbeBtagTriggerMonitor") << "B-Jet handle not valid, will skip event \n";
      return;
    }

    const reco::JetTagCollection& bTags = *(bjetHandle.product());
    for (const auto& i_jetTag : bTags) {
      const auto& jetRef = i_jetTag.first;
      const auto btagVal = i_jetTag.second;
      if (not std::isfinite(btagVal)) {
        continue;
      }
      if (allJetBTagVals.find(jetRef) != allJetBTagVals.end()) {
        allJetBTagVals.at(jetRef) += btagVal;
      } else {
        allJetBTagVals.insert(JetTagMap::value_type(jetRef, btagVal));
      }
    }
  }

  // applying selection for event; tag & probe -> selection  for all events
  if (genTriggerEventFlag_->on() && genTriggerEventFlag_->accept(iEvent, iSetup)) {
    if (allJetBTagVals.size() > 1) {
      auto jetBTagVal = allJetBTagVals.begin();
      auto jet1 = *dynamic_cast<const reco::Jet*>(jetBTagVal->first.get());
      auto btag1 = jetBTagVal->second;

      ++jetBTagVal;
      auto jet2 = *dynamic_cast<const reco::Jet*>(jetBTagVal->first.get());
      auto btag2 = jetBTagVal->second;

      if (jet1.pt() > jetPtmin_ && jet2.pt() > jetPtmin_ && fabs(jet1.eta()) < jetEtamax_ &&
          fabs(jet2.eta()) < jetEtamax_) {
        if (btag1 > tagBtagmin_ && btag2 > probeBtagmin_) {
          pt_jet1_->Fill(jet1.pt());
          pt_jet2_->Fill(jet2.pt());
          eta_jet1_->Fill(jet1.eta());
          eta_jet2_->Fill(jet2.eta());
          phi_jet1_->Fill(jet1.phi());
          phi_jet2_->Fill(jet2.phi());
          eta_phi_jet1_->Fill(jet1.eta(), jet1.phi());
          eta_phi_jet2_->Fill(jet2.eta(), jet2.phi());
          discr_offline_btag_jet1_->Fill(btag1);
          discr_offline_btag_jet2_->Fill(btag2);

          // trigger objects matching
          std::vector<trigger::TriggerObject> onlinebtags;
          edm::Handle<trigger::TriggerEvent> triggerEventHandler;
          iEvent.getByToken(triggerSummaryToken_, triggerEventHandler);
          const unsigned int filterIndex(
              triggerEventHandler->filterIndex(edm::InputTag(triggerobjbtag_, "", processname_)));
          if (filterIndex < triggerEventHandler->sizeFilters()) {
            const trigger::Keys& keys(triggerEventHandler->filterKeys(filterIndex));
            const trigger::TriggerObjectCollection& triggerObjects = triggerEventHandler->getObjects();
            for (auto& key : keys) {
              onlinebtags.reserve(onlinebtags.size() + keys.size());
              onlinebtags.push_back(triggerObjects[key]);
            }
          }
          for (auto const& to : onlinebtags) {
            if (reco::deltaR2(jet1, to) < 0.09)
              match1 = true;
            if (reco::deltaR2(jet2, to) < 0.09)
              match2 = true;
          }

          if (match1)  // jet1 is the tag
          {
            pt_probe_->Fill(jet2.pt());
            eta_probe_->Fill(jet2.eta());
            phi_probe_->Fill(jet2.phi());
            eta_phi_probe_->Fill(jet2.eta(), jet2.phi());
            if (match2)  // jet2 is the probe
            {
              pt_probe_match_->Fill(jet2.pt());
              eta_probe_match_->Fill(jet2.eta());
              phi_probe_match_->Fill(jet2.phi());
              eta_phi_probe_match_->Fill(jet2.eta(), jet2.phi());
            }
          }
        }
      }
    }
  }
}

DEFINE_FWK_MODULE(TagAndProbeBtagTriggerMonitor);
