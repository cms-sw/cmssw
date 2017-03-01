#ifndef HLTDisplacedtktkVtxProducer_h
#define HLTDisplacedtktkVtxProducer_h

/** \class HLTDisplacedtktkVtxProducer_h
 *
 *  
 *  produces kalman vertices from di-track
 *  takes track candidates as input
 *  configurable cuts on pt, eta, pair pt, inv. mass
 *  the two tracks have to be both tracks or both muons
 *
 *  \author Alexander.Schmidt@cern.ch
 *  \date   3. Feb. 2011
 *  \adapted for D Gian.Michele.Innocenti@cern.ch
 *  \date   5. Aug. 2015
 *
 */



#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include <vector>

namespace edm {
  class ConfigurationDescriptions;
}

class HLTDisplacedtktkVtxProducer : public edm::stream::EDProducer<> {
 public:
  explicit HLTDisplacedtktkVtxProducer(const edm::ParameterSet&);
  ~HLTDisplacedtktkVtxProducer();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);  
  virtual void beginJob();
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob();

 private:  
  bool checkPreviousCand(const reco::TrackRef& trackref, std::vector<reco::RecoChargedCandidateRef>& ref2);

  const edm::InputTag                                          srcTag_;
  const edm::EDGetTokenT<reco::RecoChargedCandidateCollection> srcToken_;
  const edm::InputTag                                          previousCandTag_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> previousCandToken_;
  const double maxEta_;
  const double minPt_;
  const double minPtPair_;
  const double minInvMass_;
  const double maxInvMass_;
  const double massParticle1_;
  const double massParticle2_;
  const int chargeOpt_;
  const int triggerTypeDaughters_;  

};

#endif
