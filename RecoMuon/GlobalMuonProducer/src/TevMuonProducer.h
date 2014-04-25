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
 *
 *   \author  R.Bellan - INFN TO
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonRefitter.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
// Input and output collection

#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/Common/interface/ValueMap.h"

typedef edm::ValueMap<reco::DYTInfo> DYTestimators;

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
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  
 private:
    
  /// STA Label
  edm::InputTag theGLBCollectionLabel;
  edm::EDGetTokenT<reco::TrackCollection> glbMuonsToken;
  edm::EDGetTokenT<std::vector<Trajectory> > glbMuonsTrajToken;
  


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
