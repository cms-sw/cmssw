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

TrackExtractor::TrackExtractor( const ParameterSet& par ) :
  theTrackCollectionLabel(par.getUntrackedParameter<string>("TrackCollectionLabel")),
  theDepositLabel(par.getUntrackedParameter<string>("DepositLabel")),
  theDiff_r(par.getParameter<double>("Diff_r")),
  theDiff_z(par.getParameter<double>("Diff_z")),
  theDR_Max(par.getParameter<double>("theDR_Max")),
  theDR_Veto(par.getParameter<double>("theDR_Veto"))
{
}

void TrackExtractor::fillVetos (const edm::Event & ev, const edm::EventSetup & evSetup, const reco::TrackCollection & muons) {

  theVetoCollection.clear();

  TrackCollection::const_iterator tk;
  for (tk = muons.begin(); tk != muons.end(); tk++) {
    theVetoCollection.push_back(&(*tk));
  }

}

MuIsoDeposit TrackExtractor::deposit(const Event & event, const EventSetup & eventSetup, const Track & muon) const
{
  static std::string metname = "RecoMuon/TrackExtractor";

  double vtx_z = muon.z();
  Direction muonDir(muon.eta(), muon.phi0());

  Handle<TrackCollection> tracksH;
  event.getByLabel(theTrackCollectionLabel, tracksH);
  const TrackCollection tracks = *(tracksH.product());
  LogTrace(metname)<<"***** TRACK COLLECTION SIZE: "<<tracks.size();

  TrackSelector selection(TrackSelector::Range(vtx_z-theDiff_z, vtx_z+theDiff_z),
       theDiff_r, muonDir, theDR_Max);

  TrackCollection sel_tracks = selection(tracks);
  LogTrace(metname)<<"all tracks: "<<tracks.size()<<" selected: "<<sel_tracks.size();

  MuIsoDeposit dep(theDepositLabel, muonDir.eta(), muonDir.phi() );
  
  Direction depDir(dep.getEta(), dep.getPhi());
  TrackCollection::const_iterator tk;
  for (tk = sel_tracks.begin(); tk != sel_tracks.end(); tk++) {
    if (std::find(theVetoCollection.begin(), theVetoCollection.end(), &(*tk))!=theVetoCollection.end()) continue;

    Direction dirTrk(tk->eta(), tk->phi0());
    float dr = depDir.deltaR(dirTrk);
    dep.addDeposit(dr, tk->pt());
  }

  return dep;
}
