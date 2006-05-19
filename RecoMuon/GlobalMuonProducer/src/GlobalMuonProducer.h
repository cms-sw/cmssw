#ifndef RecoMuon_GlobalMuonProducer_H
#define RecoMuon_GlobalMuonProducer_H

/**  \class GlobalMuonProducer
 * 
 *   Global muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from internal seeds (muon track segments).
 *
 *
 *   $Date: 2006/04/26 07:02:57 $
 *   $Revision: 1.1 $
 *
 *   \author  R.Bellan - INFN TO
 */

#include "FWCore/Framework/interface/EDProducer.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonTrackFinder;

class GlobalMuonProducer : public edm::EDProducer {

 public:

  /// constructor with config
  GlobalMuonProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~GlobalMuonProducer(); 
  
  /// reconstruct muons
  virtual void produce(edm::Event&, const edm::EventSetup&);
  // ex virtual void reconstruct();
  
 private:
    
  // Seed Collection Label
  std::string theSeedCollectionLabel;
  
  MuonTrackFinder* theTrackFinder; //It isn't the same as in ORCA
 
};

#endif
