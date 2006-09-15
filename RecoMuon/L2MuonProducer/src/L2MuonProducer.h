#ifndef RecoMuon_L2MuonProducer_L2MuonProducer_H
#define RecoMuon_L2MuonProducer_L2MuonProducer_H

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
 *   $Date: 2006/08/30 12:27:55 $
 *   $Revision: 1.3 $
 *
 *   \author  R.Bellan - INFN TO
 */
//
//--------------------------------------------------

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonTrackFinder;
class MuonServiceProxy;

class L2MuonProducer : public edm::EDProducer {

  public:

  /// constructor with config
  L2MuonProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~L2MuonProducer(); 
  
  /// reconstruct muons
  virtual void produce(edm::Event&, const edm::EventSetup&);
    
 private:

  // MuonSeed Collection Label
  edm::InputTag theSeedCollectionLabel;

  /// the track finder
  MuonTrackFinder* theTrackFinder; //It isn't the same as in ORCA

  /// the event setup proxy, it takes care the services update
  MuonServiceProxy *theService;
};

#endif
