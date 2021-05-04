/** \class L1TPFTauFilter
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *  \author Sandeep Bhowmik 
 *  \author Thiago Tomei
 *
 */

#include "L1TPFTauFilter.h"
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

L1TPFTauFilter::L1TPFTauFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      l1PFTauTag_(iConfig.getParameter<edm::InputTag>("inputTag")),
      pfTauToken_(consumes<l1t::PFTauCollection>(l1PFTauTag_)) {
  min_Pt_ = iConfig.getParameter<double>("MinPt");
  min_N_ = iConfig.getParameter<int>("MinN");
  min_Eta_ = iConfig.getParameter<double>("MinEta");
  max_Eta_ = iConfig.getParameter<double>("MaxEta");
  maxChargedIso_ = iConfig.getParameter<double>("MaxChargedIso");
  maxFullIso_ = iConfig.getParameter<double>("MaxFullIso");
  passLooseNN_ = iConfig.getParameter<int>("PassLooseNN");
  passTightNN_ = iConfig.getParameter<int>("PassTightNN");
  scalings_ = iConfig.getParameter<edm::ParameterSet>("Scalings");
  barrelScalings_ = scalings_.getParameter<std::vector<double> >("barrel");
  endcapScalings_ = scalings_.getParameter<std::vector<double> >("endcap");
}

L1TPFTauFilter::~L1TPFTauFilter() = default;

//
// member functions
//

void L1TPFTauFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<double>("MinPt", -1.0);
  desc.add<double>("MinEta", -5.0);
  desc.add<double>("MaxEta", 5.0);
  desc.add<int>("MinN", 1);
  desc.add<double>("MaxChargedIso", 1.0E9);  // could use std::numeric_limits<double>::max(), but it is overkill...
  desc.add<double>("MaxFullIso", 1.0E9);
  desc.add<int>("PassLooseNN", -1);
  desc.add<int>("PassTightNN", -1);
  desc.add<edm::InputTag>("inputTag", edm::InputTag("L1PFTausNN"));

  edm::ParameterSetDescription descScalings;
  descScalings.add<std::vector<double> >("barrel", {0.0, 1.0, 0.0});
  descScalings.add<std::vector<double> >("overlap", {0.0, 1.0, 0.0});
  descScalings.add<std::vector<double> >("endcap", {0.0, 1.0, 0.0});
  desc.add<edm::ParameterSetDescription>("Scalings", descScalings);

  descriptions.add("L1TPFTauFilter", desc);
}

// ------------ method called to produce the data  ------------
bool L1TPFTauFilter::hltFilter(edm::Event& iEvent,
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
    filterproduct.addCollectionTag(l1PFTauTag_);
  }

  // Specific filter code

  // get hold of products from Event
  Handle<l1t::PFTauCollection> PFTaus;
  iEvent.getByToken(pfTauToken_, PFTaus);

  // pftau
  int npftau(0);
  auto apftaus(PFTaus->begin());
  auto opftaus(PFTaus->end());
  l1t::PFTauCollection::const_iterator iPFTau;
  for (iPFTau = apftaus; iPFTau != opftaus; iPFTau++) {
    double offlinePt = this->PFTauOfflineEt(iPFTau->pt(), iPFTau->eta());
    if (offlinePt >= min_Pt_ && iPFTau->eta() <= max_Eta_ && iPFTau->eta() >= min_Eta_ &&
        iPFTau->passLooseNN() > passLooseNN_ && iPFTau->passTightNN() > passTightNN_ &&
        iPFTau->chargedIso() < maxChargedIso_ && iPFTau->fullIso() < maxFullIso_) {
      npftau++;
      l1t::PFTauRef ref(l1t::PFTauRef(PFTaus, distance(apftaus, iPFTau)));
      filterproduct.addObject(trigger::TriggerObjectType::TriggerL1PFTau, ref);
    }
  }

  // return with final filter decision
  const bool accept(npftau >= min_N_);
  return accept;
}

double L1TPFTauFilter::PFTauOfflineEt(double Et, double Eta) const {
  if (std::abs(Eta) < 1.5)
    return (barrelScalings_.at(0) + Et * barrelScalings_.at(1) + Et * Et * barrelScalings_.at(2));
  else
    return (endcapScalings_.at(0) + Et * endcapScalings_.at(1) + Et * Et * endcapScalings_.at(2));
}
