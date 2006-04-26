//-------------------------------------------------
//
/**  \class L2MuonProducer
 * 
 *   L2 muon reconstructor:
 *   reconstructs muons using DT, CSC and RPC
 *   information,<BR>
 *   starting from internal seeds (L2 muon track segments).
 *
 *
 *   $Date: 2006/03/24 13:42:50 $
 *   $Revision: 1.1 $
 *
 *   \author  R.Bellan - INFN TO
 */
//
//--------------------------------------------------

#ifndef RecoMuon_L2MuonProducer_H
#define RecoMuon_L2MuonProducer_H

#include "FWCore/Framework/interface/EDProducer.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonTrackFinder;

class L2MuonProducer : public edm::EDProducer {

  public:

  /// constructor with config
  L2MuonProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~L2MuonProducer(); 
  
  /// reconstruct muons
  virtual void produce(edm::Event&, const edm::EventSetup&);
  // ex virtual void reconstruct();
  
 private:
    
  MuonTrackFinder* theTrackFinder; //It isn't the same as in ORCA
 
};

#endif
