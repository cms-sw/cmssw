#ifndef FastSimulation_Muons_FastL3MuonProducer__H
#define FastSimulation_Muons_FastL3MuonProducer_H

/**  \class FastL3MuonProducer
 * 
 *   Fast L3 muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from a L2 reonstructed muon and from pixel seeded
 *   tracks (by default - the latter is configuratble)
 *
 *   $Date: 2008/03/14 19:12:07 $
 *   $Revision: 1.2 $
 *   \author  P. Janot - CERN
 */

#include "FWCore/Framework/interface/EDProducer.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonServiceProxy;
class MuonTrackFinder;
class FastL3MuonTrajectoryBuilder;

class FastL3MuonProducer : public edm::EDProducer {

 public:

  /// constructor with config
  FastL3MuonProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~FastL3MuonProducer(); 
  
  /// reconstruct muons
  virtual void produce(edm::Event&, const edm::EventSetup&);

  
 private:
    
  /// Seed STA  (input)
  edm::InputTag theL2CollectionLabel;

  /// Tracker tracks (input)
  edm::InputTag theTrackerTrackCollectionLabel;

  /// Label for L2SeededTracks (output)
  std::string theL2SeededTkLabel;

  // The muon track finder (from STA and tracks)
  MuonTrackFinder* theTrackFinder;
  FastL3MuonTrajectoryBuilder* l3mtb;

  bool theL2TrajectoryFlag;
  bool updatedAtVtx;

  /// the event setup proxy, it takes care the services update
  MuonServiceProxy *theService;
    
};

#endif
