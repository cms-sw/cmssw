#ifndef MuonSeedGenerator_RPCSeedGenerator_H
#define MuonSeedGenerator_RPCSeedGenerator_H

 /*  
 *  \class RPCSeedGenerator
 *  Muodule for RPC seed production.
 *
 *  $Date: 2007/3/12 08:55:24 $
 *  $Revision: 1.1 $
 *  \author D. Pagano - University of Pavia & INFN Pavia
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TFile.h"
#include <vector>

namespace edm {class ParameterSet; class Event; class EventSetup;}


class RPCSeedFinder;

class RPCSeedGenerator: public edm::EDProducer {
 public:

  // Constructor
  RPCSeedGenerator(const edm::ParameterSet&);
  
  // Destructor
  virtual ~RPCSeedGenerator();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

 protected:

 private:
  void complete(RPCSeedFinder& seed, MuonTransientTrackingRecHit::MuonRecHitContainer &recHits, bool* used=0) const;
  void checkAndFill(RPCSeedFinder& Theseed, const edm::EventSetup& eSetup);

  std::vector<TrajectorySeed> theSeeds;

  edm::InputTag theRPCRecHits;

};
#endif

