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

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//

L1TTkMuonFilter::L1TTkMuonFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      l1TkMuonTag_(iConfig.getParameter<edm::InputTag>("inputTag")),
      tkMuonToken_(consumes<TkMuonCollection>(l1TkMuonTag_)) {
  min_Pt_ = iConfig.getParameter<double>("MinPt");
  min_N_ = iConfig.getParameter<int>("MinN");
  min_Eta_ = iConfig.getParameter<double>("MinEta");
  max_Eta_ = iConfig.getParameter<double>("MaxEta");
  scalings_ = iConfig.getParameter<edm::ParameterSet>("Scalings");
  barrelScalings_ = scalings_.getParameter<std::vector<double> >("barrel");
  overlapScalings_ = scalings_.getParameter<std::vector<double> >("overlap");
  endcapScalings_ = scalings_.getParameter<std::vector<double> >("endcap");
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

  edm::ParameterSetDescription descScalings;
  descScalings.add<std::vector<double> >("barrel", {0.0, 1.0, 0.0});
  descScalings.add<std::vector<double> >("overlap", {0.0, 1.0, 0.0});
  descScalings.add<std::vector<double> >("endcap", {0.0, 1.0, 0.0});
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
  Handle<l1t::TkMuonCollection> tkMuons;
  iEvent.getByToken(tkMuonToken_, tkMuons);

  // trkMuon
  int ntrkmuon(0);
  auto atrkmuons(tkMuons->begin());
  auto otrkmuons(tkMuons->end());
  TkMuonCollection::const_iterator itkMuon;
  for (itkMuon = atrkmuons; itkMuon != otrkmuons; itkMuon++) {
    double offlinePt = this->TkMuonOfflineEt(itkMuon->pt(), itkMuon->eta());
    if (offlinePt >= min_Pt_ && itkMuon->eta() <= max_Eta_ && itkMuon->eta() >= min_Eta_) {
      ntrkmuon++;
      l1t::TkMuonRef ref(l1t::TkMuonRef(tkMuons, distance(atrkmuons, itkMuon)));
      filterproduct.addObject(trigger::TriggerObjectType::TriggerL1TkMu, ref);
    }
  }

  // return with final filter decision
  const bool accept(ntrkmuon >= min_N_);
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
