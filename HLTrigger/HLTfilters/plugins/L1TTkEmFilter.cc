/** \class L1TTkEmFilter
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *  \author Simone Gennai
 *  \author Thiago Tomei
 *
 */

#include "L1TTkEmFilter.h"
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

L1TTkEmFilter::L1TTkEmFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      l1TkEmTag1_(iConfig.getParameter<edm::InputTag>("inputTag1")),
      l1TkEmTag2_(iConfig.getParameter<edm::InputTag>("inputTag2")),
      tkEmToken1_(consumes<TkEmCollection>(l1TkEmTag1_)),
      tkEmToken2_(consumes<TkEmCollection>(l1TkEmTag2_)) {
  min_Pt_ = iConfig.getParameter<double>("MinPt");
  min_N_ = iConfig.getParameter<int>("MinN");
  min_AbsEta1_ = iConfig.getParameter<double>("MinAbsEta1");
  max_AbsEta1_ = iConfig.getParameter<double>("MaxAbsEta1");
  min_AbsEta2_ = iConfig.getParameter<double>("MinAbsEta2");
  max_AbsEta2_ = iConfig.getParameter<double>("MaxAbsEta2");
  scalings_ = iConfig.getParameter<edm::ParameterSet>("Scalings");
  barrelScalings_ = scalings_.getParameter<std::vector<double> >("barrel");
  endcapScalings_ = scalings_.getParameter<std::vector<double> >("endcap");
  etaBinsForIsolation_ = iConfig.getParameter<std::vector<double> >("EtaBinsForIsolation");
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

L1TTkEmFilter::~L1TTkEmFilter() = default;

//
// member functions
//

void L1TTkEmFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<double>("MinPt", -1.0);
  desc.add<double>("MinAbsEta1", 0.0);
  desc.add<double>("MaxAbsEta1", 1.479);
  desc.add<double>("MinAbsEta2", 1.479);
  desc.add<double>("MaxAbsEta2", 5.0);
  desc.add<int>("MinN", 1);
  desc.add<edm::InputTag>("inputTag1", edm::InputTag("L1TkEms1"));
  desc.add<edm::InputTag>("inputTag2", edm::InputTag("L1TkEms2"));
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

  descriptions.add("L1TTkEmFilter", desc);
}

// ------------ method called to produce the data  ------------
bool L1TTkEmFilter::hltFilter(edm::Event& iEvent,
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
    filterproduct.addCollectionTag(l1TkEmTag1_);
    filterproduct.addCollectionTag(l1TkEmTag2_);
  }

  // Specific filter code

  // get hold of products from Event

  /// Barrel collection
  Handle<l1t::TkEmCollection> tkEms1;
  iEvent.getByToken(tkEmToken1_, tkEms1);

  /// Endcap collection
  Handle<l1t::TkEmCollection> tkEms2;
  iEvent.getByToken(tkEmToken2_, tkEms2);

  int ntrkEm(0);
  // Loop over first collection
  auto atrkEms(tkEms1->begin());
  auto otrkEms(tkEms1->end());
  TkEmCollection::const_iterator itkEm;
  for (itkEm = atrkEms; itkEm != otrkEms; itkEm++) {
    double offlinePt = this->TkEmOfflineEt(itkEm->pt(), itkEm->eta());
    bool passQuality(false);
    bool passIsolation(false);

    if (applyQual1_) {
      if (qual1IsMask_) {
        passQuality = (itkEm->hwQual() & quality1_);
      } else {
        passQuality = (itkEm->hwQual() == quality1_);
      }
    } else
      passQuality = true;

    // There has to be a better way to do this.
    for (unsigned int etabin = 1; etabin != etaBinsForIsolation_.size(); ++etabin) {
      if (std::abs(itkEm->eta()) < etaBinsForIsolation_.at(etabin) and
          std::abs(itkEm->eta()) > etaBinsForIsolation_.at(etabin - 1) and
          itkEm->trkIsol() < trkIsolation_.at(etabin - 1))
        passIsolation = true;
    }

    if (offlinePt >= min_Pt_ && std::abs(itkEm->eta()) < max_AbsEta1_ && std::abs(itkEm->eta()) >= min_AbsEta1_ &&
        passQuality && passIsolation) {
      ntrkEm++;
      l1t::TkEmRef ref1(l1t::TkEmRef(tkEms1, distance(atrkEms, itkEm)));
      filterproduct.addObject(trigger::TriggerObjectType::TriggerL1TkEm, ref1);
    }
  }

  // Loop over second collection. Notice we don't reset ntrkEm
  atrkEms = tkEms2->begin();
  otrkEms = tkEms2->end();
  for (itkEm = atrkEms; itkEm != otrkEms; itkEm++) {
    double offlinePt = this->TkEmOfflineEt(itkEm->pt(), itkEm->eta());
    bool passQuality(false);
    bool passIsolation(false);

    if (applyQual2_) {
      if (qual2IsMask_) {
        passQuality = (itkEm->hwQual() & quality2_);
      } else {
        passQuality = (itkEm->hwQual() == quality2_);
      }
    } else
      passQuality = true;

    for (unsigned int etabin = 1; etabin != etaBinsForIsolation_.size(); ++etabin) {
      if (std::abs(itkEm->eta()) < etaBinsForIsolation_.at(etabin) and
          std::abs(itkEm->eta()) > etaBinsForIsolation_.at(etabin - 1) and
          itkEm->trkIsol() < trkIsolation_.at(etabin - 1))
        passIsolation = true;
    }

    if (offlinePt >= min_Pt_ && std::abs(itkEm->eta()) <= max_AbsEta2_ && std::abs(itkEm->eta()) >= min_AbsEta2_ &&
        passQuality && passIsolation) {
      ntrkEm++;
      l1t::TkEmRef ref2(l1t::TkEmRef(tkEms2, distance(atrkEms, itkEm)));
      filterproduct.addObject(trigger::TriggerObjectType::TriggerL1TkEm, ref2);
    }
  }

  // return with final filter decision
  const bool accept(ntrkEm >= min_N_);
  return accept;
}

double L1TTkEmFilter::TkEmOfflineEt(double Et, double Eta) const {
  if (std::abs(Eta) < 1.5)
    return (barrelScalings_.at(0) + Et * barrelScalings_.at(1) + Et * Et * barrelScalings_.at(2));
  else
    return (endcapScalings_.at(0) + Et * endcapScalings_.at(1) + Et * Et * endcapScalings_.at(2));
}
