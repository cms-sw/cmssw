#ifndef RecoMuon_L2MuonSeedGenerator_L2MuonSeedGenerator_H
#define RecoMuon_L2MuonSeedGenerator_L2MuonSeedGenerator_H

//-------------------------------------------------
//
/**  \class L2MuonSeedGenerator
 * 
 *   L2 muon seed generator:
 *   Transform the L1 informations in seeds for the
 *   L2 muon reconstruction
 *
 *
 *
 *   \author  A.Everett, R.Bellan
 *
 *    ORCA's author: N. Neumeister 
 */
//
//--------------------------------------------------

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

// Data Formats 
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
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

#include "CLHEP/Vector/ThreeVector.h"

#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

class MuonServiceProxy;
class MeasurementEstimator;
class TrajectorySeed;
class TrajectoryStateOnSurface;

namespace edm {class ParameterSet; class Event; class EventSetup;}

class L2MuonSeedGenerator : public edm::EDProducer {
 
 public:
  
  /// Constructor
  explicit L2MuonSeedGenerator(const edm::ParameterSet&);

  /// Destructor
  ~L2MuonSeedGenerator();

  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:

  edm::InputTag theSource;
  edm::InputTag theL1GMTReadoutCollection;
  edm::InputTag theOfflineSeedLabel;
  std::string   thePropagatorName;

  edm::EDGetTokenT<L1MuGMTReadoutCollection> gmtToken_;
  edm::EDGetTokenT<l1extra::L1MuonParticleCollection> muCollToken_;
  edm::EDGetTokenT<edm::View<TrajectorySeed> > offlineSeedToken_;

  const double theL1MinPt;
  const double theL1MaxEta;
  const unsigned theL1MinQuality;
  const bool useOfflineSeed;
  const bool useUnassociatedL1;

  /// the event setup proxy, it takes care the services update
  MuonServiceProxy *theService;  

  MeasurementEstimator *theEstimator;

  const TrajectorySeed* associateOfflineSeedToL1( edm::Handle<edm::View<TrajectorySeed> > &, 
						  std::vector<int> &, 
						  TrajectoryStateOnSurface &);

};

#endif
