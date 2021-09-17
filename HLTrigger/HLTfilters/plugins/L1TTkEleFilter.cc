/** \class L1TTkEleFilter
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *  \author Simone Gennai
 *  \author Thiago Tomei
 *
 */

#include "L1TTkEleFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//

L1TTkEleFilter::L1TTkEleFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      l1TkEleTag1_(iConfig.getParameter<edm::InputTag>("inputTag1")),
      l1TkEleTag2_(iConfig.getParameter<edm::InputTag>("inputTag2")),
      tkEleToken1_(consumes<TkEleCollection>(l1TkEleTag1_)),
      tkEleToken2_(consumes<TkEleCollection>(l1TkEleTag2_)) {
  min_Pt_ = iConfig.getParameter<double>("MinPt");
  min_N_ = iConfig.getParameter<int>("MinN");
  min_Eta_ = iConfig.getParameter<double>("MinEta");
  max_Eta_ = iConfig.getParameter<double>("MaxEta");
  scalings_ = iConfig.getParameter<edm::ParameterSet>("Scalings");
  barrelScalings_ = scalings_.getParameter<std::vector<double> >("barrel");
  endcapScalings_ = scalings_.getParameter<std::vector<double> >("endcap");
  etaBinsForIsolation_ = iConfig.getParameter<std::vector<double> >("EtaBinsForIsolation");
  trkIsolation_ = iConfig.getParameter<std::vector<double> >("TrkIsolation");
  quality1_ = iConfig.getParameter<int>("Quality1");
  quality2_ = iConfig.getParameter<int>("Quality2");
  qual1IsMask_ = iConfig.getParameter<bool>("Qual1IsMask");
  qual2IsMask_ = iConfig.getParameter<bool>("Qual2IsMask");
  applyQual1_ = iConfig.getParameter<bool>("ApplyQual1");
  applyQual2_ = iConfig.getParameter<bool>("ApplyQual2");

  if (etaBinsForIsolation_.size() != (trkIsolation_.size() + 1))
    throw cms::Exception("ConfigurationError")
        << "Vector of isolation values should have same size of vector of eta bins plus one.";
}

L1TTkEleFilter::~L1TTkEleFilter() = default;

//
// member functions
//

void L1TTkEleFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  makeHLTFilterDescription(desc);
  desc.add<double>("MinPt", -1.0);
  desc.add<double>("MinEta", -5.0);
  desc.add<double>("MaxEta", 5.0);
  desc.add<int>("MinN", 1);
  desc.add<edm::InputTag>("inputTag1", edm::InputTag("L1TkElectrons1"));
  desc.add<edm::InputTag>("inputTag2", edm::InputTag("L1TkElectrons2"));
  desc.add<std::vector<double> >("EtaBinsForIsolation", {0.0, 1.479});
  desc.add<std::vector<double> >("TrkIsolation",
                                 {
                                     99999.9,
                                 });
  desc.add<int>("Quality1", 0);
  desc.add<int>("Quality2", 0);
  desc.add<bool>("Qual1IsMask", false);
  desc.add<bool>("Qual2IsMask", false);
  desc.add<bool>("ApplyQual1", false);
  desc.add<bool>("ApplyQual2", false);

  edm::ParameterSetDescription descScalings;
  descScalings.add<std::vector<double> >("barrel", {0.0, 1.0, 0.0});
  descScalings.add<std::vector<double> >("endcap", {0.0, 1.0, 0.0});
  desc.add<edm::ParameterSetDescription>("Scalings", descScalings);

  descriptions.add("L1TTkEleFilter", desc);
}

// ------------ method called to produce the data  ------------
bool L1TTkEleFilter::hltFilter(edm::Event& iEvent,
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
    filterproduct.addCollectionTag(l1TkEleTag1_);
    filterproduct.addCollectionTag(l1TkEleTag2_);
  }

  // Specific filter code

  // get hold of products from Event

  /// Barrel colleciton
  Handle<l1t::TkElectronCollection> tkEles1;
  iEvent.getByToken(tkEleToken1_, tkEles1);

  /// Endcap collection
  Handle<l1t::TkElectronCollection> tkEles2;
  iEvent.getByToken(tkEleToken2_, tkEles2);

  int ntrkEle(0);
  // Loop over first collection
  auto atrkEles(tkEles1->begin());
  auto otrkEles(tkEles1->end());
  TkEleCollection::const_iterator itkEle;
  for (itkEle = atrkEles; itkEle != otrkEles; itkEle++) {
    double offlinePt = this->TkEleOfflineEt(itkEle->pt(), itkEle->eta());
    bool passQuality(false);
    bool passIsolation(false);

    if (applyQual1_) {
      if (qual1IsMask_)
        passQuality = (itkEle->EGRef()->hwQual() & quality1_);
      else
        passQuality = (itkEle->EGRef()->hwQual() == quality1_);
    } else
      passQuality = true;

    // There has to be a better way to do this.
    for (unsigned int etabin = 1; etabin != etaBinsForIsolation_.size(); ++etabin) {
      if (std::abs(itkEle->eta()) < etaBinsForIsolation_.at(etabin) and
          std::abs(itkEle->eta()) > etaBinsForIsolation_.at(etabin - 1) and
          itkEle->trkIsol() < trkIsolation_.at(etabin - 1))
        passIsolation = true;
    }

    if (offlinePt >= min_Pt_ && itkEle->eta() <= max_Eta_ && itkEle->eta() >= min_Eta_ && passQuality &&
        passIsolation) {
      ntrkEle++;
      l1t::TkElectronRef ref1(l1t::TkElectronRef(tkEles1, distance(atrkEles, itkEle)));
      filterproduct.addObject(trigger::TriggerObjectType::TriggerL1TkEle, ref1);
    }
  }

  // Loop over second collection. Notice we don't reset ntrkEle
  atrkEles = tkEles2->begin();
  otrkEles = tkEles2->end();
  for (itkEle = atrkEles; itkEle != otrkEles; itkEle++) {
    double offlinePt = this->TkEleOfflineEt(itkEle->pt(), itkEle->eta());
    bool passQuality(false);
    bool passIsolation(false);

    if (applyQual2_) {
      if (qual2IsMask_)
        passQuality = (itkEle->EGRef()->hwQual() & quality2_);
      else
        passQuality = (itkEle->EGRef()->hwQual() == quality2_);
    } else
      passQuality = true;

    for (unsigned int etabin = 1; etabin != etaBinsForIsolation_.size(); ++etabin) {
      if (std::abs(itkEle->eta()) < etaBinsForIsolation_.at(etabin) and
          std::abs(itkEle->eta()) > etaBinsForIsolation_.at(etabin - 1) and
          itkEle->trkIsol() < trkIsolation_.at(etabin - 1))
        passIsolation = true;
    }

    if (offlinePt >= min_Pt_ && itkEle->eta() <= max_Eta_ && itkEle->eta() >= min_Eta_ && passQuality &&
        passIsolation) {
      ntrkEle++;
      l1t::TkElectronRef ref2(l1t::TkElectronRef(tkEles2, distance(atrkEles, itkEle)));
      filterproduct.addObject(trigger::TriggerObjectType::TriggerL1TkEle, ref2);
    }
  }

  // return with final filter decision
  const bool accept(ntrkEle >= min_N_);
  return accept;
}

double L1TTkEleFilter::TkEleOfflineEt(double Et, double Eta) const {
  if (std::abs(Eta) < 1.5)
    return (barrelScalings_.at(0) + Et * barrelScalings_.at(1) + Et * Et * barrelScalings_.at(2));
  else
    return (endcapScalings_.at(0) + Et * endcapScalings_.at(1) + Et * Et * endcapScalings_.at(2));
}
