#ifndef RecoMuon_CosmicMuonProducer_GlobalCosmicMuonProducer_H
#define RecoMuon_CosmicMuonProducer_GlobalCosmicMuonProducer_H

/** \file CosmicMuonProducer
 *
 *  reconstruct muons using dt,csc,rpc and tracker starting from cosmic muon 
 *  tracks
 *
 *  $Date: $
 *  $Revision: $
 *  \author Chang Liu  -  Purdue University <Chang.Liu@cern.ch>
 */

#include "FWCore/Framework/interface/EDProducer.h"

class MuonTrackFinder;
class MuonServiceProxy;

class GlobalCosmicMuonProducer : public edm::EDProducer {
public:
  explicit GlobalCosmicMuonProducer(const edm::ParameterSet&);

   ~GlobalCosmicMuonProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:
  std::string theTrackCollectionLabel;
  MuonTrackFinder* theTrackFinder;

  /// the event setup proxy, it takes care the services update
  MuonServiceProxy *theService;
};

#endif
