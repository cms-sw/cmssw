#include "RecoMuon/MuonIsolation/src/TrackExtractor.h"

#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "RecoMuon/MuonIsolation/interface/Direction.h"
#include "RecoMuon/MuonIsolation/src/TrackSelector.h"
#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;

TrackExtractor::TrackExtractor( double aDdiff_r, double aDiff_z, double aDR_match, double aDR_Veto, 
  std::string aTrackCollectionLabel, std::string aDepositLabel) 
  : theDiff_r(aDdiff_r), theDiff_z(aDiff_z), theDR_Match(aDR_match), theDR_Veto(aDR_Veto),
    theTrackCollectionLabel(aTrackCollectionLabel), theDepositLabel(aDepositLabel) 
{ }


vector<MuIsoDeposit> TrackExtractor::deposits( const Event & event, 
    const EventSetup & eventSetup, const Track & muon, 
    const vector<Direction> & vetoDirections, double coneSize) const
{
  vector<MuIsoDeposit> result;
  static std::string metname = "RecoMuon/TrackExtractor";

  double vtx_z = muon.z();
  Direction muonDir(muon.eta(), muon.phi0());

  Handle<TrackCollection> tracksH;
  event.getByLabel(theTrackCollectionLabel, tracksH);
  const TrackCollection tracks = *(tracksH.product());
  LogTrace(metname)<<"***** TRACK COLLECTION SIZE: "<<tracks.size();

  TrackSelector selection( TrackSelector::Range(vtx_z-theDiff_z, vtx_z+theDiff_z),
       theDiff_r, muonDir, coneSize);

  TrackCollection selected_tracks = selection(tracks);
  LogTrace(metname)<<"all tracks: "<<tracks.size()<<" selected: "<<selected_tracks.size();

  MuIsoDeposit deposit(theDepositLabel, muonDir.eta(), muonDir.phi() );
  fillDeposits(deposit, selected_tracks, vetoDirections);

  result.push_back(deposit);

  return result;
}
  

void TrackExtractor::fillDeposits( MuIsoDeposit & deposit, 
    const TrackCollection & tracks, const std::vector<Direction> & vetos) const
{
  Direction depDir(deposit.getEta(), deposit.getPhi());
  for (TrackCollection::const_iterator it = tracks.begin(); it != tracks.end(); it++) {
    Direction dirTrk(it->eta(), it->phi0());
    bool skip = false;
    for (vector<Direction>::const_iterator iv = vetos.begin(); iv != vetos.end(); iv++) {
      if( dirTrk.deltaR(*iv) < theDR_Veto) { skip = true; break; }
    }
    if (skip) continue;
    float dr = depDir.deltaR(dirTrk);
    deposit.addDeposit(dr, it->pt());
  }
}
