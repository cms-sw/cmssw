#include "PixelTrackExtractor.h"

#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "TrackSelector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"


using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;
using reco::isodeposit::Direction;

PixelTrackExtractor::PixelTrackExtractor( const ParameterSet& par, edm::ConsumesCollector && iC ) :
  theTrackCollectionToken(iC.consumes<TrackCollection >(par.getParameter<edm::InputTag>("inputTrackCollection"))),
  theDepositLabel(par.getUntrackedParameter<string>("DepositLabel")),
  theDiff_r(par.getParameter<double>("Diff_r")),
  theDiff_z(par.getParameter<double>("Diff_z")),
  theDR_Max(par.getParameter<double>("DR_Max")),
  theDR_Veto(par.getParameter<double>("DR_Veto")),
  theBeamlineOption(par.getParameter<string>("BeamlineOption")),
  theBeamSpotToken(iC.mayConsume<BeamSpot>(par.getParameter<edm::InputTag>("BeamSpotLabel"))),
  theNHits_Min(par.getParameter<unsigned int>("NHits_Min")),
  theChi2Ndof_Max(par.getParameter<double>("Chi2Ndof_Max")),
  theChi2Prob_Min(par.getParameter<double>("Chi2Prob_Min")),
  thePt_Min(par.getParameter<double>("Pt_Min")),
  thePropagateTracksToRadius(par.getParameter<bool>("PropagateTracksToRadius")),
  theReferenceRadius(par.getParameter<double>("ReferenceRadius")),
  theVetoLeadingTrack(par.getParameter<bool>("VetoLeadingTrack")), //! will veto leading track if
  thePtVeto_Min(par.getParameter<double>("PtVeto_Min")),           //! .. it is above this threshold
  theDR_VetoPt(par.getParameter<double>("DR_VetoPt"))              //!.. and is inside this cone
{
}

reco::IsoDeposit::Vetos PixelTrackExtractor::vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Track & track) const
{
  Direction dir(track.eta(),track.phi());
  return reco::IsoDeposit::Vetos(1,veto(dir));
}

reco::IsoDeposit::Veto PixelTrackExtractor::veto(const Direction & dir) const
{
  reco::IsoDeposit::Veto result;
  result.vetoDir = dir;
  result.dR = theDR_Veto;
  return result;
}

Direction PixelTrackExtractor::directionAtPresetRadius(const Track& tk, double bz) const {
  if (! thePropagateTracksToRadius ){
    return Direction(tk.eta(), tk.phi());
  }

  // this should represent a cylinder in global frame at R=refRadius cm, roughly where mid-layer of pixels is
  double psRadius = theReferenceRadius;
  double tkDxy = tk.dxy();
  double s2D = fabs(tk.dxy()) < psRadius ? sqrt(psRadius*psRadius - tkDxy*tkDxy) : 0;

  // the field we get from the caller is already in units of GeV/cm
  double dPhi = -s2D*tk.charge()*bz/tk.pt();

  return Direction(tk.eta(), tk.phi()+dPhi);
}

IsoDeposit PixelTrackExtractor::deposit(const Event & event, const EventSetup & eventSetup, const Track & muon) const
{
  static const std::string metname = "MuonIsolation|PixelTrackExtractor";

  edm::ESHandle<MagneticField> bField;
  eventSetup.get<IdealMagneticFieldRecord>().get(bField);
  double bz = bField->inInverseGeV(GlobalPoint(0.,0.,0.)).z();

  Direction muonDir(directionAtPresetRadius(muon, bz));
  IsoDeposit deposit(muonDir );
  //! Note, this can be reset below if theVetoLeadingTrack is set and conditions are met
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


  double maxPt = -1;
  Direction maxPtDir;
  TrackSelector::result_type::const_iterator tkI = sel_tracks.begin();
  for (; tkI != sel_tracks.end(); ++tkI) {
    const reco::Track* tk = *tkI;
    LogTrace(metname) << "This track has: pt= " << tk->pt() << ", eta= "
        << tk->eta() <<", phi= "<<tk->phi();
    Direction dirTrk(directionAtPresetRadius(*tk, bz));
    deposit.addDeposit(dirTrk, tk->pt());
    double tkDr = (muonDir-dirTrk).deltaR;
    double tkPt = tk->pt();
    if (theVetoLeadingTrack && tkPt > thePtVeto_Min
	&& tkDr < theDR_VetoPt
	&& maxPt < tkPt ){
      maxPt = tkPt;
      maxPtDir = dirTrk;
    }
  }
  if (maxPt > 0){
    deposit.setVeto(veto(maxPtDir));
    LogTrace(metname)<<" Set track veto the leading track with pt "
		     <<maxPt<<" in direction  (eta,phi) "
		     <<maxPtDir.eta()<<", "<<maxPtDir.phi();
  }

  return deposit;
}
