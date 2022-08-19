/** \class L1TTkMuonFilter
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *  \author Simone Gennai
 *  \author Thiago Tomei
 *
 */

#include "L1TTkMuonFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//

namespace {
  bool isDupMuon(const l1t::TrackerMuonRef& muon, const std::vector<l1t::TrackerMuonRef>& existing) {
    for (const auto& exist : existing) {
      //it is our understanding that there is an exact eta phi match
      //and we should not be concerned with numerical precision
      // DISCUSS: should this use hardware or physics pt?
      // For the time being we use hardware pt, because this is just for a duplicate check.
      if (reco::deltaR2(muon->hwEta(), muon->hwPhi(), exist->hwEta(), exist->hwPhi()) <= 0) {
        return true;
      }
    }
    return false;
  }
}  // namespace

L1TTkMuonFilter::L1TTkMuonFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      l1TkMuonTag_(iConfig.getParameter<edm::InputTag>("inputTag")),
      tkMuonToken_(consumes<l1t::TrackerMuonCollection>(l1TkMuonTag_)),
      qualityCut_(iConfig.getParameter<edm::ParameterSet>("qualities")) {
  min_Pt_ = iConfig.getParameter<double>("MinPt");
  min_N_ = iConfig.getParameter<int>("MinN");
  min_Eta_ = iConfig.getParameter<double>("MinEta");
  max_Eta_ = iConfig.getParameter<double>("MaxEta");
  applyQuality_ = iConfig.getParameter<bool>("applyQuality");
  applyDuplicateRemoval_ = iConfig.getParameter<bool>("applyDuplicateRemoval");
  scalings_ = iConfig.getParameter<edm::ParameterSet>("Scalings");
  barrelScalings_ = scalings_.getParameter<std::vector<double>>("barrel");
  overlapScalings_ = scalings_.getParameter<std::vector<double>>("overlap");
  endcapScalings_ = scalings_.getParameter<std::vector<double>>("endcap");
}

L1TTkMuonFilter::~L1TTkMuonFilter() = default;

//
// member functions
//

void L1TTkMuonFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<double>("MinPt", -1.0);
  desc.add<double>("MinEta", -5.0);
  desc.add<double>("MaxEta", 5.0);
  desc.add<int>("MinN", 1);
  desc.add<edm::InputTag>("inputTag", edm::InputTag("L1TkMuons"));
  desc.add<bool>("applyQuality", false);
  desc.add<bool>("applyDuplicateRemoval", true);
  desc.add<edm::ParameterSetDescription>("qualities", MuonQualityCut::makePSetDescription());

  edm::ParameterSetDescription descScalings;
  descScalings.add<std::vector<double>>("barrel", {0.0, 1.0, 0.0});
  descScalings.add<std::vector<double>>("overlap", {0.0, 1.0, 0.0});
  descScalings.add<std::vector<double>>("endcap", {0.0, 1.0, 0.0});
  desc.add<edm::ParameterSetDescription>("Scalings", descScalings);

  descriptions.add("L1TTkMuonFilter", desc);
}

// ------------ method called to produce the data  ------------
bool L1TTkMuonFilter::hltFilter(edm::Event& iEvent,
                                const edm::EventSetup& iSetup,
                                trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  // The filter object
  if (saveTags()) {
    filterproduct.addCollectionTag(l1TkMuonTag_);
  }

  // Specific filter code

  // get hold of products from Event
  Handle<l1t::TrackerMuonCollection> tkMuons;
  iEvent.getByToken(tkMuonToken_, tkMuons);

  //it looks rather slow to get the added muons back out of the filterproduct
  //so we just make a vector of passing and then add them all at the end
  std::vector<l1t::TrackerMuonRef> passingMuons;
  auto atrkmuons(tkMuons->begin());
  auto otrkmuons(tkMuons->end());
  l1t::TrackerMuonCollection::const_iterator itkMuon;
  for (itkMuon = atrkmuons; itkMuon != otrkmuons; itkMuon++) {
    double offlinePt = this->TkMuonOfflineEt(itkMuon->phPt(), itkMuon->phEta());
    bool passesQual = !applyQuality_ || qualityCut_(*itkMuon);
    if (passesQual && offlinePt >= min_Pt_ && itkMuon->phEta() <= max_Eta_ && itkMuon->phEta() >= min_Eta_) {
      l1t::TrackerMuonRef ref(tkMuons, distance(atrkmuons, itkMuon));
      if (!applyDuplicateRemoval_ || !isDupMuon(ref, passingMuons)) {
        passingMuons.push_back(ref);
      }
    }
  }
  for (const auto& muon : passingMuons) {
    filterproduct.addObject(trigger::TriggerObjectType::TriggerL1TkMu, muon);
  }

  // return with final filter decision
  const bool accept(static_cast<int>(passingMuons.size()) >= min_N_);
  return accept;
}

L1TTkMuonFilter::MuonQualityCut::MuonQualityCut(const edm::ParameterSet& iConfig) {
  auto detQualities = iConfig.getParameter<std::vector<edm::ParameterSet>>("values");
  for (const auto& detQuality : detQualities) {
    auto dets = detQuality.getParameter<std::vector<int>>("detectors");
    auto qualities = detQuality.getParameter<std::vector<int>>("qualities");
    std::sort(qualities.begin(), qualities.end());
    for (const auto& det : dets) {
      allowedQualities_.insert({det, std::move(qualities)});
    }
  }
}

bool L1TTkMuonFilter::MuonQualityCut::operator()(const l1t::TrackerMuon& muon) const {
  // If we didn't load any qualities from the config file, that means we don't care.
  // So we set passesQuality to true.
  if (allowedQualities_.empty()) {
    return true;
  }

  bool passesQuality(false);
  // The muonDetector() method is not available for TrackerMuon (it was available in TkMuon),
  // so for the time being we just take the union of all quality vectors for all muon detectors.
  for (const auto& detQuality : allowedQualities_) {
    passesQuality |= std::binary_search(detQuality.second.begin(), detQuality.second.end(), muon.hwQual());
  }

  return passesQuality;
}

void L1TTkMuonFilter::MuonQualityCut::fillPSetDescription(edm::ParameterSetDescription& desc) {
  edm::ParameterSetDescription detQualDesc;
  detQualDesc.add<std::vector<int>>("detectors", {-99});
  detQualDesc.add<std::vector<int>>("qualities", {-99, -99, -99});
  std::vector<edm::ParameterSet> detQualDefaults;
  edm::ParameterSet detQualDefault;
  detQualDefault.addParameter<std::vector<int>>("detectors", {-99});
  detQualDefault.addParameter<std::vector<int>>("qualities", {-99, -99, -99});
  detQualDefaults.push_back(detQualDefault);
  desc.addVPSet("values", detQualDesc, detQualDefaults);
}

edm::ParameterSetDescription L1TTkMuonFilter::MuonQualityCut::makePSetDescription() {
  edm::ParameterSetDescription desc;
  fillPSetDescription(desc);
  return desc;
}

double L1TTkMuonFilter::TkMuonOfflineEt(double Et, double Eta) const {
  if (std::abs(Eta) < 0.9)
    return (barrelScalings_.at(0) + Et * barrelScalings_.at(1) + Et * Et * barrelScalings_.at(2));
  else if (std::abs(Eta) < 1.2)
    return (overlapScalings_.at(0) + Et * overlapScalings_.at(1) + Et * Et * overlapScalings_.at(2));
  else
    return (endcapScalings_.at(0) + Et * endcapScalings_.at(1) + Et * Et * endcapScalings_.at(2));
}
