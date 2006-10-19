#ifndef RecoMuon_StandAloneMuonProducer_StandAloneMuonProducer_H
#define RecoMuon_StandAloneMuonProducer_StandAloneMuonProducer_H

/**  \class StandAloneMuonProducer
 * 
 *   StandAlone muon reconstructor:
 *   reconstructs muons using DT, CSC and RPC
 *   information,<BR>
 *   starting from internal seeds (L2 muon track segments).
 *
 *
 *   $Date: 2006/09/15 12:04:31 $
 *   $Revision: 1.5 $
 *
 *   \author  R.Bellan - INFN TO
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonTrackFinder;
class MuonServiceProxy;

class StandAloneMuonProducer : public edm::EDProducer {

 public:

  /// constructor with config
  StandAloneMuonProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~StandAloneMuonProducer(); 
  
  /// reconstruct muons
  virtual void produce(edm::Event&, const edm::EventSetup&);
    
 private:
  
  /// MuonSeed Collection Label
  edm::InputTag theSeedCollectionLabel;
 
  /// Put Trajectory into Event Flag
  bool theTrajectoryFlag;
  
  /// the track finder
  MuonTrackFinder* theTrackFinder; //It isn't the same as in ORCA

  /// the event setup proxy, it takes care the services update
  MuonServiceProxy *theService;
};

#endif
