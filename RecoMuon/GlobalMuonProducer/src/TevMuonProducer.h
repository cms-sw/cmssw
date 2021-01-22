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

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
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

class MuonTrackFinder;
class MuonServiceProxy;
class TrackerTopologyRcd;

class TevMuonProducer : public edm::stream::EDProducer<> {
public:
  /// constructor with config
  TevMuonProducer(const edm::ParameterSet&);

  /// destructor
  ~TevMuonProducer() override;

  /// reconstruct muons
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  /// STA Label
  edm::InputTag theGLBCollectionLabel;
  edm::EDGetTokenT<reco::TrackCollection> glbMuonsToken;
  edm::EDGetTokenT<std::vector<Trajectory> > glbMuonsTrajToken;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken;

  /// the event setup proxy, it takes care the services update
  std::unique_ptr<MuonServiceProxy> theService;

  std::unique_ptr<GlobalMuonRefitter> theRefitter;

  std::unique_ptr<MuonTrackLoader> theTrackLoader;

  std::string theAlias;
  std::vector<std::string> theRefits;
  std::vector<int> theRefitIndex;

  void setAlias(std::string alias) {
    alias.erase(alias.size() - 1, alias.size());
    theAlias = alias;
  }
};

#endif
