#ifndef RecoMuon_L3MuonProducer_L3MuonProducer_H
#define RecoMuon_L3MuonProducer_L3MuonProducer_H

/**  \class L3MuonProducer
 * 
 *   L3 muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from a L2 reonstructed muon.
 *
 *   $Date: 2007/01/03 16:25:28 $
 *   $Revision: 1.2 $
 *   \author  A. Everett - Purdue University
 */

#include "FWCore/Framework/interface/EDProducer.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonTrackFinder;
class MuonServiceProxy;

class L3MuonProducer : public edm::EDProducer {

 public:

  /// constructor with config
  L3MuonProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~L3MuonProducer(); 
  
  /// reconstruct muons
  virtual void produce(edm::Event&, const edm::EventSetup&);

  
 private:
    
  /// Seed STA Label
  edm::InputTag theL2CollectionLabel;

  /// Label for L2SeededTracks
  std::string theL2SeededTkLabel; 

  bool theL2TrajectoryFlag;

  MuonTrackFinder* theTrackFinder;
    
  /// the event setup proxy, it takes care the services update
  MuonServiceProxy *theService;
    
};

#endif
