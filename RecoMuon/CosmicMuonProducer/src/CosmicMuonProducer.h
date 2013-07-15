#ifndef RecoMuon_CosmicMuonProducer_CosmicMuonProducer_H
#define RecoMuon_CosmicMuonProducer_CosmicMuonProducer_H

/** \file CosmicMuonProducer
 *
 *  $Date: 2006/07/03 01:11:19 $
 *  $Revision: 1.2 $
 *  \author Chang Liu
 */

#include "FWCore/Framework/interface/EDProducer.h"

class MuonTrackFinder;
class MuonServiceProxy;

class CosmicMuonProducer : public edm::EDProducer {
public:
  explicit CosmicMuonProducer(const edm::ParameterSet&);

   ~CosmicMuonProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:
  std::string theSeedCollectionLabel;
  MuonTrackFinder* theTrackFinder;

  /// the event setup proxy, it takes care the services update
  MuonServiceProxy *theService;
};

#endif
