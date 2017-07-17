#include "TrackExtractor.h"

#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "TrackSelector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;
using reco::isodeposit::Direction;

TrackExtractor::TrackExtractor( const ParameterSet& par, edm::ConsumesCollector && iC ) :
  theTrackCollectionToken(iC.consumes<TrackCollection>(par.getParameter<edm::InputTag>("inputTrackCollection"))),
  theDepositLabel(par.getUntrackedParameter<string>("DepositLabel")),
  theDiff_r(par.getParameter<double>("Diff_r")),
  theDiff_z(par.getParameter<double>("Diff_z")),
  theDR_Max(par.getParameter<double>("DR_Max")),
  theDR_Veto(par.getParameter<double>("DR_Veto")),
  theBeamlineOption(par.getParameter<string>("BeamlineOption")),
  theBeamSpotToken(iC.consumes<reco::BeamSpot>(par.getParameter<edm::InputTag>("BeamSpotLabel"))),
  theNHits_Min(par.getParameter<unsigned int>("NHits_Min")),
  theChi2Ndof_Max(par.getParameter<double>("Chi2Ndof_Max")),
  theChi2Prob_Min(par.getParameter<double>("Chi2Prob_Min")),
  thePt_Min(par.getParameter<double>("Pt_Min"))
{
}

reco::IsoDeposit::Vetos TrackExtractor::vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Track & track) const
{
  reco::isodeposit::Direction dir(track.eta(),track.phi());
  return reco::IsoDeposit::Vetos(1,veto(dir));
}

reco::IsoDeposit::Veto TrackExtractor::veto(const reco::IsoDeposit::Direction & dir) const
{
  reco::IsoDeposit::Veto result;
  result.vetoDir = dir;
  result.dR = theDR_Veto;
  return result;
}

IsoDeposit TrackExtractor::deposit(const Event & event, const EventSetup & eventSetup, const Track & muon) const
{
  static const std::string metname = "MuonIsolation|TrackExtractor";

  reco::isodeposit::Direction muonDir(muon.eta(), muon.phi());
  IsoDeposit deposit(muonDir );
  deposit.setVeto( veto(muonDir) );
  deposit.addCandEnergy(muon.pt());

  Handle<TrackCollection> tracksH;
  event.getByToken(theTrackCollectionToken, tracksH);
  //  const TrackCollection tracks = *(tracksH.product());
  LogTrace(metname)<<"***** TRACK COLLECTION SIZE: "<<tracksH->size();

  double vtx_z = muon.vz();
  LogTrace(metname)<<"***** Muon vz: "<<vtx_z;
  reco::TrackBase::Point beamPoint(0,0, 0);

  if (theBeamlineOption == "BeamSpotFromEvent"){
    //pick beamSpot
    reco::BeamSpot beamSpot;
    edm::Handle<reco::BeamSpot> beamSpotH;

    event.getByToken(theBeamSpotToken,beamSpotH);

    if (beamSpotH.isValid()){
      beamPoint = beamSpotH->position();
      LogTrace(metname)<<"Extracted beam point at "<<beamPoint<<std::endl;
    }
  }

  LogTrace(metname)<<"Using beam point at "<<beamPoint<<std::endl;

  TrackSelector::Parameters pars(TrackSelector::Range(vtx_z-theDiff_z, vtx_z+theDiff_z),
				 theDiff_r, muonDir, theDR_Max, beamPoint);

  pars.nHitsMin = theNHits_Min;
  pars.chi2NdofMax = theChi2Ndof_Max;
  pars.chi2ProbMin = theChi2Prob_Min;
  pars.ptMin = thePt_Min;

  TrackSelector selection(pars);
  TrackSelector::result_type sel_tracks = selection(*tracksH);
  LogTrace(metname)<<"all tracks: "<<tracksH->size()<<" selected: "<<sel_tracks.size();


  TrackSelector::result_type::const_iterator tkI = sel_tracks.begin();
  for (; tkI != sel_tracks.end(); ++tkI) {
    const reco::Track* tk = *tkI;
    LogTrace(metname) << "This track has: pt= " << tk->pt() << ", eta= "
        << tk->eta() <<", phi= "<<tk->phi();
    reco::isodeposit::Direction dirTrk(tk->eta(), tk->phi());
    deposit.addDeposit(dirTrk, tk->pt());
  }

  return deposit;
}
