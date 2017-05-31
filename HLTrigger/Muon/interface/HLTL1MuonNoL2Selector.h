#ifndef HLTTrigger_HLTL1MuonNoL2Selector_HLTL1MuonNoL2Selector_H
#define HLTTrigger_HLTL1MuonNoL2Selector_HLTL1MuonNoL2Selector_H

//-------------------------------------------------
//
/**  \class HLTL1MuonNoL2Selector
 * 
 *   HLTL1MuonNoL2Selector:
 *   Simple selector to output a subset of L1 muon collection 
 *   
 *   based on RecoMuon/L2MuonSeedGenerator
 *
 *
 *   \author  S. Folgueras
 */
//
//--------------------------------------------------

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

// Data Formats 
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "HLTrigger/Muon/interface/HLTMuonL2ToL1TMap.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class HLTL1MuonNoL2Selector : public edm::global::EDProducer<> {
 
 public:
  
  /// Constructor
  explicit HLTL1MuonNoL2Selector(const edm::ParameterSet&);

  /// Destructor
  ~HLTL1MuonNoL2Selector();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  
 private:

  edm::InputTag theL1Source_;
  const double theL1MinPt_;
  const double theL1MaxEta_;
  const unsigned theL1MinQuality_;
  bool centralBxOnly_;

  edm::EDGetTokenT<l1t::MuonBxCollection> muCollToken_;
  edm::InputTag                                          theL2CandTag_;   
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> theL2CandToken_;
  edm::InputTag             seedMapTag_;
  edm::EDGetTokenT<SeedMap> seedMapToken_;
  


};

#endif
