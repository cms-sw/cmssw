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
 *   $Date: 2012/01/24 10:58:53 $
 *   $Revision: 1.7 $
 *
 *   \author  A.Everett, R.Bellan
 *
 *    ORCA's author: N. Neumeister 
 */
//
//--------------------------------------------------

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

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

  const double theL1MinPt;
  const double theL1MaxEta;
  const unsigned theL1MinQuality;
  const bool useOfflineSeed;

  /// the event setup proxy, it takes care the services update
  MuonServiceProxy *theService;  

  MeasurementEstimator *theEstimator;

  const TrajectorySeed* associateOfflineSeedToL1( edm::Handle<edm::View<TrajectorySeed> > &, 
						  std::vector<int> &, 
						  TrajectoryStateOnSurface &);

};

#endif
