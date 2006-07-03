#ifndef CosmicMuonProducer_h
#define CosmicMuonProducer_h

/** \file CosmicMuonProducer
 *
 *  $Date: 2006/06/14 00:07:07 $
 *  $Revision: 1.1 $
 *  \author Chang Liu
 */

#include "FWCore/Framework/interface/EDProducer.h"

class CosmicMuonTrajectoryBuilder;
class MuonTrajectoryCleaner;
class MuonTrackFinder;

class CosmicMuonProducer : public edm::EDProducer {
public:
  explicit CosmicMuonProducer(const edm::ParameterSet&);

   ~CosmicMuonProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:
  std::string theSeedCollectionLabel;
  MuonTrackFinder* theTrackFinder;

};

#endif
