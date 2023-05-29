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
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
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
      tkMuonToken_(consumes(l1TkMuonTag_)) {
  min_Pt_ = iConfig.getParameter<double>("MinPt");
  min_N_ = iConfig.getParameter<int>("MinN");
  min_Eta_ = iConfig.getParameter<double>("MinEta");
  max_Eta_ = iConfig.getParameter<double>("MaxEta");
  applyQuality_ = iConfig.getParameter<bool>("applyQuality");
  applyDuplicateRemoval_ = iConfig.getParameter<bool>("applyDuplicateRemoval");
  qualities_ = iConfig.getParameter<std::vector<int>>("qualities");
  scalings_ = iConfig.getParameter<edm::ParameterSet>("Scalings");
  barrelScalings_ = scalings_.getParameter<std::vector<double>>("barrel");
  overlapScalings_ = scalings_.getParameter<std::vector<double>>("overlap");
  endcapScalings_ = scalings_.getParameter<std::vector<double>>("endcap");

  std::sort(qualities_.begin(), qualities_.end());

  if (applyQuality_ && qualities_.empty()) {
    throw cms::Exception("InvalidConfiguration")
        << "If you want to applyQuality the qualities vector should not be empty!";
  }
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
  desc.add<std::vector<int>>("qualities", {});
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

    // The muonDetector() and quality() methods are not available for TrackerMuon
    // as they were for TkMuon (TkMuon being the old implementation, at the times
    // of the HLT-TDR). So we fall back to the hwQual() method inherited from
    // L1Candidate, and compare it with a vector of allowed qualities.
    bool passesQual = !applyQuality_ || std::binary_search(qualities_.begin(), qualities_.end(), itkMuon->hwQual());

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

double L1TTkMuonFilter::TkMuonOfflineEt(double Et, double Eta) const {
  if (std::abs(Eta) < 0.9)
    return (barrelScalings_.at(0) + Et * barrelScalings_.at(1) + Et * Et * barrelScalings_.at(2));
  else if (std::abs(Eta) < 1.2)
    return (overlapScalings_.at(0) + Et * overlapScalings_.at(1) + Et * Et * overlapScalings_.at(2));
  else
    return (endcapScalings_.at(0) + Et * endcapScalings_.at(1) + Et * Et * endcapScalings_.at(2));
}
