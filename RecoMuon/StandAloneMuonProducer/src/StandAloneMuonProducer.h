#ifndef RecoMuon_StandAloneMuonProducer_H
#define RecoMuon_StandAloneMuonProducer_H

/**  \class StandAloneMuonProducer
 * 
 *   StandAlone muon reconstructor:
 *   reconstructs muons using DT, CSC and RPC
 *   information,<BR>
 *   starting from internal seeds (L2 muon track segments).
 *
 *
 *   $Date: 2006/04/26 07:00:55 $
 *   $Revision: 1.1 $
 *
 *   \author  R.Bellan - INFN TO
 */

#include "FWCore/Framework/interface/EDProducer.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonTrackFinder;

class StandAloneMuonProducer : public edm::EDProducer {

 public:

  /// constructor with config
  StandAloneMuonProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~StandAloneMuonProducer(); 
  
  /// reconstruct muons
  virtual void produce(edm::Event&, const edm::EventSetup&);
  // ex virtual void reconstruct();
  
 private:
  
  // MuonSeed Collection Label
  std::string theMuonSeedLabel;
  
  MuonTrackFinder* theTrackFinder; //It isn't the same as in ORCA
};

#endif
