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
  theDR_Max(par.getParameter<double>("DR_Max")),
  theDR_Veto(par.getParameter<double>("DR_Veto"))
{
}

void TrackExtractor::fillVetos (const edm::Event & ev, const edm::EventSetup & evSetup, const reco::TrackCollection & muons) {
  static std::string metname = "RecoMuon/TrackExtractor";

  theVetoCollection = &muons;
  for (unsigned int i=0; i<theVetoCollection->size(); i++) {
    Track mu = theVetoCollection->at(i);
    LogTrace(metname) << "Track to veto: pt= " << mu.pt() << ", eta= " 
        << mu.eta() <<", phi= "<<mu.phi0();
  }

  /*
  theVetoCollection.clear();
  for (unsigned int i=0; i<muons.size(); i++) {
    Track mu = muons[i];
    theVetoCollection.push_back(mu);
    LogTrace(metname) << "Track to veto: pt= " << mu.pt() << ", eta= " 
        << mu.eta() <<", phi= "<<mu.phi0();
  }
  */

}

MuIsoDeposit TrackExtractor::deposit(const Event & event, const EventSetup & eventSetup, const Track & muon) const
{
  static std::string metname = "RecoMuon/TrackExtractor";

  double vtx_z = muon.vz();
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
  
  Direction depDir(dep.eta(), dep.phi());
  TrackCollection::const_iterator tk;
  for (tk = sel_tracks.begin(); tk != sel_tracks.end(); tk++) {
    LogTrace(metname) << "This track has: pt= " << tk->pt() << ", eta= " 
        << tk->eta() <<", phi= "<<tk->phi0();
    bool veto_this_track = false;
    Direction dirTrk(tk->eta(), tk->phi0());
    LogTrace(metname) << "From direction: eta= " << dirTrk.eta() << ", phi= "<<dirTrk.phi();
    for (unsigned int i=0; i<theVetoCollection->size(); i++) {
            const Track tkveto = theVetoCollection->at(i);
            Direction vetoDir(tkveto.eta(), tkveto.phi0());
            float drveto = dirTrk.deltaR(vetoDir);
            LogTrace(metname) << "Veto track has: pt= " << tkveto.pt() << ", eta= " << tkveto.eta() <<", phi= "<<tkveto.phi0();
            LogTrace(metname) << "From direction: eta= " << vetoDir.eta() << ", phi= "<<vetoDir.phi();
            LogTrace(metname) << "iveto= " << i <<", drveto= " << drveto;
            if (drveto<theDR_Veto) {
                  veto_this_track = true;
                  break;
            }
    }
    if (veto_this_track) continue;
    LogTrace(metname) << "This track is in the deposit cone: pt= " << tk->pt();

    float dr = depDir.deltaR(dirTrk);
    dep.addDeposit(dr, tk->pt());
  }

  return dep;
}
