#ifndef RecoMuon_GlobalMuonProducer_GlobalMuonProducer_H
#define RecoMuon_GlobalMuonProducer_GlobalMuonProducer_H

/**  \class GlobalMuonProducer
 * 
 *   Global muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from a standalone reonstructed muon.
 *
 *
 *   $Date: 2006/07/06 12:35:11 $
 *   $Revision: 1.3 $
 *
 *   \author  R.Bellan - INFN TO
 */

#include "FWCore/Framework/interface/EDProducer.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonTrackFinder;
class MuonServiceProxy;

class GlobalMuonProducer : public edm::EDProducer {

 public:

  /// constructor with config
  GlobalMuonProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~GlobalMuonProducer(); 
  
  /// reconstruct muons
  virtual void produce(edm::Event&, const edm::EventSetup&);

  
 private:
    
  /// Seed STA Label
  std::string theSTACollectionLabel;
  
  MuonTrackFinder* theTrackFinder;
    
  /// the event setup proxy, it takes care the services update
  MuonServiceProxy *theService;
    
};

#endif
