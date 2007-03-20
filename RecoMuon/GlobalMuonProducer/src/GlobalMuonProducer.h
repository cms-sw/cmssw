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
 *   $Date: 2006/12/10 21:54:54 $
 *   $Revision: 1.8 $
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
    
  /// STA Label
  edm::InputTag theSTACollectionLabel;

  bool theSTATrajectoryFlag;
  
  MuonTrackFinder* theTrackFinder;
    
  /// the event setup proxy, it takes care the services update
  MuonServiceProxy* theService;
    
  std::string alias_;

  void setAlias( std::string alias ){
    alias.erase( alias.size() - 1, alias.size() );
    alias_=alias;
  }

};

#endif
