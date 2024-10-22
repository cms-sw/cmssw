// -*- C++ -*-
//
// Package:    HLTCaloTowerFilter
// Class:      HLTCaloTowerFilter
//
/**\class HLTCaloTowerFilter HLTCaloTowerFilter.cc Work/HLTCaloTowerFilter/src/HLTCaloTowerFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
// Original Author:  Yen-Jie Lee
//         Created:  Wed Nov 13 16:12:29 CEST 2009

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDefs.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//
class HLTCaloTowerFilter : public HLTFilter {
public:
  explicit HLTCaloTowerFilter(const edm::ParameterSet&);
  ~HLTCaloTowerFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<CaloTowerCollection> inputToken_;
  edm::InputTag inputTag_;  // input tag identifying product
  double min_Pt_;           // pt threshold in GeV
  double max_Eta_;          // eta range (symmetric)
  unsigned int min_N_;      // number of objects passing cuts required
};

//
// constructors and destructor
//
HLTCaloTowerFilter::HLTCaloTowerFilter(const edm::ParameterSet& config)
    : HLTFilter(config),
      inputTag_(config.getParameter<edm::InputTag>("inputTag")),
      min_Pt_(config.getParameter<double>("MinPt")),
      max_Eta_(config.getParameter<double>("MaxEta")),
      min_N_(config.getParameter<unsigned int>("MinN")) {
  inputToken_ = consumes<CaloTowerCollection>(inputTag_);
}

HLTCaloTowerFilter::~HLTCaloTowerFilter() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void HLTCaloTowerFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag", edm::InputTag("hltTowerMakerForEcal"));
  desc.add<double>("MinPt", 3.0);
  desc.add<double>("MaxEta", 3.0);
  desc.add<unsigned int>("MinN", 1);
  descriptions.add("hltCaloTowerFilter", desc);
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool HLTCaloTowerFilter::hltFilter(edm::Event& event,
                                   const edm::EventSetup& setup,
                                   trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  using namespace std;
  using namespace edm;
  using namespace reco;

  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  // The filter object
  if (saveTags())
    filterproduct.addCollectionTag(inputTag_);

  // get hold of collection of objects
  Handle<CaloTowerCollection> caloTowers;
  event.getByToken(inputToken_, caloTowers);

  // look at all objects, check cuts and add to filter object
  unsigned int n = 0;
  for (auto const& i : *caloTowers) {
    if ((i.pt() >= min_Pt_) and ((max_Eta_ < 0.0) or (std::abs(i.eta()) <= max_Eta_)))
      ++n;
    //edm::Ref<CaloTowerCollection> ref(towers, std::distance(caloTowers->begin(), i));
    //filterproduct.addObject(TriggerJet, ref);
  }

  // filter decision
  return (n >= min_N_);
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTCaloTowerFilter);
