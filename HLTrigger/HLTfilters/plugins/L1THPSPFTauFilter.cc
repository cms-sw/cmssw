/** \class L1THPSPFTauFilter
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *  \author Sandeep Bhowmik 
 *  \author Thiago Tomei
 *
 */

#include "L1THPSPFTauFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//

L1THPSPFTauFilter::L1THPSPFTauFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      l1HPSPFTauTag_(iConfig.getParameter<edm::InputTag>("inputTag")),
      hpspfTauToken_(consumes<l1t::HPSPFTauCollection>(l1HPSPFTauTag_)) {
  min_Pt_ = iConfig.getParameter<double>("MinPt");
  min_N_ = iConfig.getParameter<int>("MinN");
  min_Eta_ = iConfig.getParameter<double>("MinEta");
  max_Eta_ = iConfig.getParameter<double>("MaxEta");
  max_RelChargedIso_ = iConfig.getParameter<double>("MaxRelChargedIso");
  min_LeadTrackPt_ = iConfig.getParameter<double>("MinLeadTrackPt");
  scalings_ = iConfig.getParameter<edm::ParameterSet>("Scalings");
  barrelScalings_ = scalings_.getParameter<std::vector<double> >("barrel");
  overlapScalings_ = scalings_.getParameter<std::vector<double> >("overlap");
  endcapScalings_ = scalings_.getParameter<std::vector<double> >("endcap");
}

L1THPSPFTauFilter::~L1THPSPFTauFilter() = default;

//
// member functions
//

void L1THPSPFTauFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<double>("MinPt", -1.0);
  desc.add<double>("MinEta", -5.0);
  desc.add<double>("MaxEta", 5.0);
  desc.add<double>("MaxRelChargedIso", 1.0E9);
  desc.add<double>("MinLeadTrackPt", -1.0);
  desc.add<int>("MinN", 1);
  desc.add<edm::InputTag>("inputTag", edm::InputTag("L1HPSPFTaus"));

  edm::ParameterSetDescription descScalings;
  descScalings.add<std::vector<double> >("barrel", {0.0, 1.0, 0.0});
  descScalings.add<std::vector<double> >("overlap", {0.0, 1.0, 0.0});
  descScalings.add<std::vector<double> >("endcap", {0.0, 1.0, 0.0});
  desc.add<edm::ParameterSetDescription>("Scalings", descScalings);

  descriptions.add("L1THPSPFTauFilter", desc);
}

// ------------ method called to produce the data  ------------
bool L1THPSPFTauFilter::hltFilter(edm::Event& iEvent,
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
    filterproduct.addCollectionTag(l1HPSPFTauTag_);
  }

  // Specific filter code

  // get hold of products from Event
  Handle<l1t::HPSPFTauCollection> HPSPFTaus;
  iEvent.getByToken(hpspfTauToken_, HPSPFTaus);

  // pftau
  int npftau(0);
  auto apftaus(HPSPFTaus->begin());
  auto opftaus(HPSPFTaus->end());
  l1t::HPSPFTauCollection::const_iterator iHPSPFTau;
  for (iHPSPFTau = apftaus; iHPSPFTau != opftaus; iHPSPFTau++) {
    double offlinePt = this->HPSPFTauOfflineEt(iHPSPFTau->pt(), iHPSPFTau->eta());
    if (offlinePt >= min_Pt_ && iHPSPFTau->eta() <= max_Eta_ && iHPSPFTau->eta() >= min_Eta_ &&
        (iHPSPFTau->sumChargedIso() / iHPSPFTau->pt()) < max_RelChargedIso_ &&
        (iHPSPFTau->leadChargedPFCand()->pfTrack()->pt()) > min_LeadTrackPt_) {
      npftau++;
      l1t::HPSPFTauRef ref(l1t::HPSPFTauRef(HPSPFTaus, distance(apftaus, iHPSPFTau)));
      filterproduct.addObject(trigger::TriggerObjectType::TriggerL1PFTau, ref);
    }
  }

  // return with final filter decision
  const bool accept(npftau >= min_N_);
  return accept;
}

double L1THPSPFTauFilter::HPSPFTauOfflineEt(double Et, double Eta) const {
  if (std::abs(Eta) < 1.5)
    return (barrelScalings_.at(0) + Et * barrelScalings_.at(1) + Et * Et * barrelScalings_.at(2));
  else
    return (endcapScalings_.at(0) + Et * endcapScalings_.at(1) + Et * Et * endcapScalings_.at(2));
}
