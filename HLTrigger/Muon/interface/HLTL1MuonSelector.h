#ifndef HLTTrigger_HLTL1MuonSelector_HLTL1MuonSelector_H
#define HLTTrigger_HLTL1MuonSelector_HLTL1MuonSelector_H

//-------------------------------------------------
//
/**  \class HLTL1MuonSelector
 * 
 *   HLTL1MuonSelector:
 *   Simple selector to output a subset of L1 muon collection 
 *   
 *   based on RecoMuon/L2MuonSeedGenerator
 *
 *
 *   \author  D. Olivito
 */
//
//--------------------------------------------------

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

// Data Formats 
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/Math/interface/deltaR.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class HLTL1MuonSelector : public edm::global::EDProducer<> {
 
 public:
  
  /// Constructor
  explicit HLTL1MuonSelector(const edm::ParameterSet&);

  /// Destructor
  ~HLTL1MuonSelector();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  
 private:

  edm::InputTag theSource;

  edm::EDGetTokenT<l1extra::L1MuonParticleCollection> muCollToken_;

  const double theL1MinPt;
  const double theL1MaxEta;
  const unsigned theL1MinQuality;

};

#endif
