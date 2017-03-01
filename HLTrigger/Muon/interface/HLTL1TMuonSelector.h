#ifndef HLTTrigger_HLTL1TMuonSelector_HLTL1TMuonSelector_H
#define HLTTrigger_HLTL1TMuonSelector_HLTL1TMuonSelector_H

//-------------------------------------------------
//
/**  \class HLTL1TMuonSelector
 * 
 *   HLTL1TMuonSelector:
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
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/Math/interface/deltaR.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class HLTL1TMuonSelector : public edm::global::EDProducer<> {
 
 public:
  
  /// Constructor
  explicit HLTL1TMuonSelector(const edm::ParameterSet&);

  /// Destructor
  ~HLTL1TMuonSelector();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  
 private:

  edm::InputTag theSource;

  edm::EDGetTokenT<l1t::MuonBxCollection> muCollToken_;

  const double theL1MinPt;
  const double theL1MaxEta;
  const unsigned theL1MinQuality;

  /// use central bx only muons
  bool centralBxOnly_;

};

#endif
