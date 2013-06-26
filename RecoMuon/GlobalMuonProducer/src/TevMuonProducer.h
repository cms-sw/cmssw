#ifndef RecoMuon_GlobalMuonProducer_TevMuonProducer_H
#define RecoMuon_GlobalMuonProducer_TevMuonProducer_H

/**  \class TevMuonProducer
 * 
 *   Global muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from a standalone reonstructed muon.
 *
 *
 *   $Date: 2008/05/13 03:31:44 $
 *   $Revision: 1.2 $
 *
 *   \author  R.Bellan - INFN TO
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonRefitter.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonTrackFinder;
class MuonServiceProxy;

class TevMuonProducer : public edm::EDProducer {

 public:

  /// constructor with config
  TevMuonProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~TevMuonProducer(); 
  
  /// reconstruct muons
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
    
  /// STA Label
  edm::InputTag theGLBCollectionLabel;

  /// the event setup proxy, it takes care the services update
  MuonServiceProxy* theService;
  
  GlobalMuonRefitter* theRefitter;

  MuonTrackLoader* theTrackLoader;
  
  std::string theAlias;
  std::vector<std::string> theRefits;
  std::vector<int> theRefitIndex;

  void setAlias( std::string alias ){
    alias.erase( alias.size() - 1, alias.size() );
    theAlias=alias;
  }
  
};

#endif
