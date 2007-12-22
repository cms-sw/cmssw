#include "TrackExtractor.h"

#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "DataFormats/MuonReco/interface/Direction.h"
#include "TrackSelector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;

TrackExtractor::TrackExtractor( const ParameterSet& par ) :
  theTrackCollectionTag(par.getParameter<edm::InputTag>("inputTrackCollection")),
  theDepositLabel(par.getUntrackedParameter<string>("DepositLabel")),
  theDiff_r(par.getParameter<double>("Diff_r")),
  theDiff_z(par.getParameter<double>("Diff_z")),
  theDR_Max(par.getParameter<double>("DR_Max")),
  theDR_Veto(par.getParameter<double>("DR_Veto"))
{
}

reco::MuIsoDeposit::Vetos TrackExtractor::vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Track & track) const
{
  Direction dir(track.eta(),track.phi());
  return reco::MuIsoDeposit::Vetos(1,veto(dir));
}

reco::MuIsoDeposit::Veto TrackExtractor::veto(const reco::MuIsoDeposit::Direction & dir) const
{
  reco::MuIsoDeposit::Veto result;
  result.vetoDir = dir;
  result.dR = theDR_Veto;
  return result;
}

MuIsoDeposit TrackExtractor::deposit(const Event & event, const EventSetup & eventSetup, const Track & muon) const
{
  static std::string metname = "MuonIsolation|TrackExtractor";

  Direction muonDir(muon.eta(), muon.phi());
  MuIsoDeposit deposit(theDepositLabel, muonDir );
  deposit.setVeto( veto(muonDir) );
  deposit.addMuonEnergy(muon.pt());

  Handle<View<Track> > tracksH;
  event.getByLabel(theTrackCollectionTag, tracksH);
  //  const TrackCollection tracks = *(tracksH.product());
  LogTrace(metname)<<"***** TRACK COLLECTION SIZE: "<<tracksH->size();

  double vtx_z = muon.vz();
  LogTrace(metname)<<"***** Muon vz: "<<vtx_z;
  TrackSelector selection(TrackSelector::Range(vtx_z-theDiff_z, vtx_z+theDiff_z),
       theDiff_r, muonDir, theDR_Max);
  TrackSelector::result_type sel_tracks = selection(*tracksH);
  LogTrace(metname)<<"all tracks: "<<tracksH->size()<<" selected: "<<sel_tracks.size();

  
  TrackSelector::result_type::const_iterator tkI = sel_tracks.begin();
  for (; tkI != sel_tracks.end(); ++tkI) {
    const reco::Track* tk = *tkI;
    LogTrace(metname) << "This track has: pt= " << tk->pt() << ", eta= " 
        << tk->eta() <<", phi= "<<tk->phi();
    Direction dirTrk(tk->eta(), tk->phi());
    deposit.addDeposit(dirTrk, tk->pt());
  }

  return deposit;
}
