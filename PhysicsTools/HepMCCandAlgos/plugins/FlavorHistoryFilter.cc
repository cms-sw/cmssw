// -*- C++ -*-
//
// Package:    FlavorHistoryFilter
// Class:      FlavorHistoryFilter
//
/**\class FlavorHistoryFilter FlavorHistoryFilter.cc PhysicsTools/FlavorHistoryFilter/src/FlavorHistoryFilter.cc

 Description:

 This now filters events hierarchically. Previously this was done at the python configuration
 level, which was cumbersome for users to use.

 Now, the hierarchy is:

 Create prioritized paths to separate HF composition samples.

 These are exclusive priorities, so sample "i" will not overlap with "i+1".
 Note that the "dr" values below correspond to the dr between the
 matched genjet, and the sister genjet.

 1) W+bb with >= 2 jets from the ME (dr > 0.5)
 2) W+b or W+bb with 1 jet from the ME
 3) W+cc from the ME (dr > 0.5)
 4) W+c or W+cc with 1 jet from the ME
 5) W+bb with 1 jet from the parton shower (dr == 0.0)
 6) W+cc with 1 jet from the parton shower (dr == 0.0)

 These are the "trash bin" samples that we're throwing away:

 7) W+bb with >= 2 partons but 1 jet from the ME (dr == 0.0)
 8) W+cc with >= 2 partons but 1 jet from the ME (dr == 0.0)
 9) W+bb with >= 2 partons but 2 jets from the PS (dr > 0.5)
 10)W+cc with >= 2 partons but 2 jets from the PS (dr > 0.5)

 And here is the true "light flavor" sample:

 11) Veto of all the previous (W+ light jets)

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  "Salvatore Rappoccio"
//         Created:  Sat Jun 28 00:41:21 CDT 2008
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HepMCCandidate/interface/FlavorHistoryEvent.h"
#include "DataFormats/HepMCCandidate/interface/FlavorHistory.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "PhysicsTools/HepMCCandAlgos/interface/FlavorHistorySelectorUtil.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "PhysicsTools/HepMCCandAlgos/interface/FlavorHistorySelectorUtil.h"

#include <vector>

//
// class declaration
//

class FlavorHistoryFilter : public edm::global::EDFilter<> {
public:
  typedef reco::FlavorHistory::FLAVOR_T flavor_type;
  typedef std::vector<int> flavor_vector;

  explicit FlavorHistoryFilter(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::StreamID, ::edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::FlavorHistoryEvent> bsrcToken_;  // Input b flavor history collection name
  edm::EDGetTokenT<reco::FlavorHistoryEvent> csrcToken_;  // Input c flavor history collection name
  int pathToSelect_;                                      // Select any of the following paths:
  double dr_;                                             // dr with which to cut off the events
  // Note! The "b" and "c" here refer to the number of matched b and c genjets, respectively
  std::unique_ptr<reco::FlavorHistorySelectorUtil const> bb_me_;  // To select bb->2 events from matrix element... Path 1
  std::unique_ptr<reco::FlavorHistorySelectorUtil const> b_me_;  // To select  b->1 events from matrix element... Path 2
  std::unique_ptr<reco::FlavorHistorySelectorUtil const> cc_me_;  // To select cc->2 events from matrix element... Path 3
  std::unique_ptr<reco::FlavorHistorySelectorUtil const> c_me_;  // To select  c->1 events from matrix element... Path 4
  std::unique_ptr<reco::FlavorHistorySelectorUtil const> b_ps_;  // To select bb->2 events from parton shower ... Path 5
  std::unique_ptr<reco::FlavorHistorySelectorUtil const> c_ps_;  // To select cc->2 events from parton shower ... Path 6
  std::unique_ptr<reco::FlavorHistorySelectorUtil const>
      bb_me_comp_;  // To select bb->1 events from matrix element... Path 7
  std::unique_ptr<reco::FlavorHistorySelectorUtil const>
      cc_me_comp_;  // To select cc->1 events from matrix element... Path 8
  std::unique_ptr<reco::FlavorHistorySelectorUtil const>
      b_ps_comp_;  // To select bb->2 events from parton shower ... Path 9
  std::unique_ptr<reco::FlavorHistorySelectorUtil const>
      c_ps_comp_;  // To select cc->1 events from parton shower ... Path 10
                   // The veto of all of these is               ... Path 11
};

using namespace edm;
using namespace reco;
using namespace std;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FlavorHistoryFilter::FlavorHistoryFilter(const edm::ParameterSet& iConfig)
    : bsrcToken_(consumes<FlavorHistoryEvent>(iConfig.getParameter<edm::InputTag>("bsrc"))),
      csrcToken_(consumes<FlavorHistoryEvent>(iConfig.getParameter<edm::InputTag>("csrc"))) {
  pathToSelect_ = iConfig.getParameter<int>("pathToSelect");

  // This is the "interface" delta R with which to decide
  // where to take the event from
  dr_ = iConfig.getParameter<double>("dr");
  bool verbose(iConfig.getParameter<bool>("verbose"));

  // Set up the boundaries.
  // dr0 = 0.0
  // dr1 = set by user
  // dr2 = infinity
  double dr0 = 0.0;
  double dr1 = dr_;
  double dr2 = 99999.0;

  // These are the processes that can come from the matrix element calculation
  std::vector<int> me_ids;
  me_ids.push_back(2);  // flavor excitation
  me_ids.push_back(3);  // flavor creation

  // These are the processes that can come from the parton shower calculation
  std::vector<int> ps_ids;
  ps_ids.push_back(1);  // gluon splitting

  // To select bb->2 events from matrix element... Path 1
  bb_me_ = std::make_unique<FlavorHistorySelectorUtil>(5, 2, me_ids, dr1, dr2, verbose);

  // To select  b->1 events from matrix element... Path 2
  b_me_ = std::make_unique<FlavorHistorySelectorUtil>(5, 1, me_ids, dr0, dr0, verbose);

  // To select cc->2 events from matrix element... Path 3
  cc_me_ = std::make_unique<FlavorHistorySelectorUtil>(4, 2, me_ids, dr1, dr2, verbose);

  // To select  c->1 events from matrix element... Path 4
  c_me_ = std::make_unique<FlavorHistorySelectorUtil>(4, 1, me_ids, dr0, dr0, verbose);

  // To select bb->2 events from parton shower ... Path 5
  b_ps_ = std::make_unique<FlavorHistorySelectorUtil>(5, 1, ps_ids, dr0, dr1, verbose);

  // To select cc->2 events from parton shower ... Path 6
  c_ps_ = std::make_unique<FlavorHistorySelectorUtil>(4, 1, ps_ids, dr0, dr1, verbose);

  // To select bb->1 events from matrix element... Path 7
  bb_me_comp_ = std::make_unique<FlavorHistorySelectorUtil>(5, 2, me_ids, dr0, dr1, verbose);

  // To select cc->1 events from matrix element... Path 8
  cc_me_comp_ = std::make_unique<FlavorHistorySelectorUtil>(4, 2, me_ids, dr0, dr1, verbose);

  // To select bb->2 events from parton shower ... Path 9
  b_ps_comp_ = std::make_unique<FlavorHistorySelectorUtil>(5, 2, ps_ids, dr1, dr2, verbose);

  // To select cc->1 events from parton shower ... Path 10
  c_ps_comp_ = std::make_unique<FlavorHistorySelectorUtil>(4, 2, ps_ids, dr1, dr2, verbose);

  // The veto of all of these is               ... Path 11

  // This will write 1-11 (the path number), or 0 if error.
  produces<unsigned int>();
}

void FlavorHistoryFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("bsrc");
  desc.add<edm::InputTag>("csrc");
  desc.add<int>("pathToSelect", -1);
  desc.add<double>("dr");
  desc.add<bool>("verbose");

  descriptions.addDefault(desc);
}

bool FlavorHistoryFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Get the flavor history
  Handle<FlavorHistoryEvent> bFlavorHistoryEvent;
  iEvent.getByToken(bsrcToken_, bFlavorHistoryEvent);

  Handle<FlavorHistoryEvent> cFlavorHistoryEvent;
  iEvent.getByToken(csrcToken_, cFlavorHistoryEvent);

  auto selection = std::make_unique<unsigned int>(0);

  // Get the number of matched b-jets in the event
  unsigned int nb = bFlavorHistoryEvent->nb();
  // Get the number of matched c-jets in the event
  unsigned int nc = cFlavorHistoryEvent->nc();
  // Get the two flavor sources. The highest takes precedence
  // over the rest.
  FlavorHistory::FLAVOR_T bFlavorSource = bFlavorHistoryEvent->flavorSource();
  FlavorHistory::FLAVOR_T cFlavorSource = cFlavorHistoryEvent->flavorSource();
  FlavorHistory::FLAVOR_T flavorSource = FlavorHistory::FLAVOR_NULL;
  // Get the highest flavor in the event
  unsigned int highestFlavor = 0;
  // Get the delta r between the two heavy flavor matched jets.
  double dr = -1;

  // Preference is in increasing priority:
  //  1: gluon splitting
  //  2: flavor excitation
  //  3: flavor creation (matrix element)
  //  4: flavor decay
  if (bFlavorSource >= cFlavorSource) {
    flavorSource = bFlavorHistoryEvent->flavorSource();
    highestFlavor = bFlavorHistoryEvent->highestFlavor();
    dr = bFlavorHistoryEvent->deltaR();
  } else {
    flavorSource = cFlavorHistoryEvent->flavorSource();
    highestFlavor = cFlavorHistoryEvent->highestFlavor();
    dr = cFlavorHistoryEvent->deltaR();
  }

  *selection = 0;
  // Now make hierarchical determination
  if (bb_me_->select(nb, nc, highestFlavor, flavorSource, dr))
    *selection = 1;
  else if (b_me_->select(nb, nc, highestFlavor, flavorSource, dr))
    *selection = 2;
  else if (cc_me_->select(nb, nc, highestFlavor, flavorSource, dr))
    *selection = 3;
  else if (c_me_->select(nb, nc, highestFlavor, flavorSource, dr))
    *selection = 4;
  else if (b_ps_->select(nb, nc, highestFlavor, flavorSource, dr))
    *selection = 5;
  else if (c_ps_->select(nb, nc, highestFlavor, flavorSource, dr))
    *selection = 6;
  else if (bb_me_comp_->select(nb, nc, highestFlavor, flavorSource, dr))
    *selection = 7;
  else if (cc_me_comp_->select(nb, nc, highestFlavor, flavorSource, dr))
    *selection = 8;
  else if (b_ps_comp_->select(nb, nc, highestFlavor, flavorSource, dr))
    *selection = 9;
  else if (c_ps_comp_->select(nb, nc, highestFlavor, flavorSource, dr))
    *selection = 10;
  else
    *selection = 11;

  bool pass = false;
  if (pathToSelect_ > 0) {
    pass = (*selection > 0 && *selection == static_cast<unsigned int>(pathToSelect_));
  } else {
    pass = true;
  }

  iEvent.put(std::move(selection));

  return pass;
}

//define this as a plug-in
DEFINE_FWK_MODULE(FlavorHistoryFilter);
