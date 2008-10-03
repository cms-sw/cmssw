#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaTrackExtractor.h"

#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaRange.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaTrackSelector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace egammaisolation;
using reco::isodeposit::Direction;

EgammaTrackExtractor::EgammaTrackExtractor( const ParameterSet& par ) :
    theTrackCollectionTag(par.getParameter<edm::InputTag>("inputTrackCollection")),
    theDepositLabel(par.getUntrackedParameter<string>("DepositLabel")),
    theDiff_r(par.getParameter<double>("Diff_r")),
    theDiff_z(par.getParameter<double>("Diff_z")),
    theDR_Max(par.getParameter<double>("DR_Max")),
    theDR_Veto(par.getParameter<double>("DR_Veto")),
    theBeamlineOption(par.getParameter<string>("BeamlineOption")),
    theBeamSpotLabel(par.getParameter<edm::InputTag>("BeamSpotLabel")),
    theNHits_Min(par.getParameter<uint>("NHits_Min")),
    theChi2Ndof_Max(par.getParameter<double>("Chi2Ndof_Max")),
    theChi2Prob_Min(par.getParameter<double>("Chi2Prob_Min")),
    thePt_Min(par.getParameter<double>("Pt_Min"))
{
}

reco::IsoDeposit::Vetos EgammaTrackExtractor::vetos(const edm::Event & ev,
        const edm::EventSetup & evSetup, const reco::Track & track) const
{
    reco::isodeposit::Direction dir(track.eta(),track.phi());
    return reco::IsoDeposit::Vetos(1,veto(dir));
}

reco::IsoDeposit::Veto EgammaTrackExtractor::veto(const reco::IsoDeposit::Direction & dir) const
{
    reco::IsoDeposit::Veto result;
    result.vetoDir = dir;
    result.dR = theDR_Veto;
    return result;
}

IsoDeposit EgammaTrackExtractor::deposit(const Event & event, const EventSetup & eventSetup, const Candidate & candTk) const
{
    static std::string metname = "EgammaIsolationAlgos|EgammaTrackExtractor";

    double vtx_z=0; 

    reco::isodeposit::Direction candDir;
    if( candTk.isElectron() ){ // center on the gsf track for electrons
      const reco::GsfElectron* elec = dynamic_cast<const reco::GsfElectron*>(&candTk);
      candDir = reco::isodeposit::Direction(elec->gsfTrack()->eta(), elec->gsfTrack()->phi());
      vtx_z = elec->gsfTrack()->dz();
    }
    else{
      candDir = reco::isodeposit::Direction(candTk.eta(), candTk.phi());
      vtx_z = candTk.vertex().z();
    }

    IsoDeposit deposit(candDir );
    deposit.setVeto( veto(candDir) );
    deposit.addCandEnergy(candTk.et());

    Handle<View<Track> > tracksH;
    event.getByLabel(theTrackCollectionTag, tracksH);

    reco::TrackBase::Point beamPoint(0,0, 0);

    if (theBeamlineOption.compare("BeamSpotFromEvent") == 0){
        //pick beamSpot
      reco::BeamSpot beamSpot;
      edm::Handle<reco::BeamSpot> beamSpotH;
      
      event.getByLabel(theBeamSpotLabel,beamSpotH);
      
      if (beamSpotH.isValid()){
	beamPoint = beamSpotH->position();	
      }
    }

    EgammaTrackSelector::Parameters pars(EgammaTrackSelector::Range(vtx_z-theDiff_z, vtx_z+theDiff_z),
					 theDiff_r, candDir, theDR_Max, beamPoint);

    pars.nHitsMin = theNHits_Min;
    pars.chi2NdofMax = theChi2Ndof_Max;
    pars.chi2ProbMin = theChi2Prob_Min;
    pars.ptMin = thePt_Min;
    
    EgammaTrackSelector selection(pars);
    EgammaTrackSelector::result_type sel_tracks = selection(*tracksH);
    LogTrace(metname)<<"all tracks: "<<tracksH->size()<<" selected: "<<sel_tracks.size();
    
    EgammaTrackSelector::result_type::const_iterator tkI = sel_tracks.begin();
    for (; tkI != sel_tracks.end(); ++tkI) {
      const reco::Track* tk = *tkI;
      reco::isodeposit::Direction dirTrk(tk->eta(), tk->phi());
      deposit.addDeposit(dirTrk, tk->pt());
    }
    
    return deposit;
}
