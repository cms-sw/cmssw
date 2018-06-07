#ifndef HLTDisplacedmumumuVtxProducer_h
#define HLTDisplacedmumumuVtxProducer_h

/** \class HLTDisplacedmumumuVtxProducer_h
 *
 *  
 *  produces kalman vertices from di-muons
 *  takes muon candidates as input
 *  configurable cuts on pt, eta, pair pt, inv. mass
 *
 *  \author Alexander.Schmidt@cern.ch
 *  \date   3. Feb. 2011
 *
 */



#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include <vector>

namespace edm {
  class ConfigurationDescriptions;
}

class HLTDisplacedmumumuVtxProducer : public edm::EDProducer {
 public:
  explicit HLTDisplacedmumumuVtxProducer(const edm::ParameterSet&);
  ~HLTDisplacedmumumuVtxProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  void beginJob() override ;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override ;

 private:  
  bool checkPreviousCand(const reco::TrackRef& trackref, std::vector<reco::RecoChargedCandidateRef>& ref2);

  edm::InputTag                                          srcTag_;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> srcToken_;
  edm::InputTag                                          previousCandTag_;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> previousCandToken_;
  double maxEta_;
  double minPt_;
  double minPtTriplet_;
  double minInvMass_;
  double maxInvMass_;
  int chargeOpt_;
};

#endif
