#ifndef RecoMuon_CosmicMuonProducer_GlobalCosmicMuonProducer_H
#define RecoMuon_CosmicMuonProducer_GlobalCosmicMuonProducer_H

/** \file CosmicMuonProducer
 *
 *  reconstruct muons using dt,csc,rpc and tracker starting from cosmic muon 
 *  tracks
 *
 *  $Date: 2010/02/11 00:14:17 $
 *  $Revision: 1.3 $
 *  \author Chang Liu  -  Purdue University <Chang.Liu@cern.ch>
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

class MuonTrackFinder;
class MuonServiceProxy;

class GlobalCosmicMuonProducer : public edm::EDProducer {
public:
  explicit GlobalCosmicMuonProducer(const edm::ParameterSet&);

   ~GlobalCosmicMuonProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:
  edm::InputTag theTrackCollectionLabel;
  MuonTrackFinder* theTrackFinder;

  /// the event setup proxy, it takes care the services update
  MuonServiceProxy *theService;
};

#endif
