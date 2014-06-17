#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaTrackExtractor.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRange.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTrackSelector.h"
#include "DataFormats/Common/interface/Handle.h"
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
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace egammaisolation;
using reco::isodeposit::Direction;

EgammaTrackExtractor::EgammaTrackExtractor( const ParameterSet& par, edm::ConsumesCollector & iC ) :
    theTrackCollectionToken(iC.consumes<View<Track> >(par.getParameter<edm::InputTag>("inputTrackCollection"))),
    theDepositLabel(par.getUntrackedParameter<std::string>("DepositLabel")),
    theDiff_r(par.getParameter<double>("Diff_r")),
    theDiff_z(par.getParameter<double>("Diff_z")),
    theDR_Max(par.getParameter<double>("DR_Max")),
    theDR_Veto(par.getParameter<double>("DR_Veto")),
    theBeamlineOption(par.getParameter<std::string>("BeamlineOption")),
    theBeamSpotToken(iC.mayConsume<reco::BeamSpot>(par.getParameter<edm::InputTag>("BeamSpotLabel"))),
    theNHits_Min(par.getParameter<unsigned int>("NHits_Min")),
    theChi2Ndof_Max(par.getParameter<double>("Chi2Ndof_Max")),
    theChi2Prob_Min(par.getParameter<double>("Chi2Prob_Min")),
    thePt_Min(par.getParameter<double>("Pt_Min")),
    dzOptionString(par.getParameter<std::string>("dzOption"))
{
    if( ! dzOptionString.compare("dz") )      dzOption = EgammaTrackSelector::dz;
    else if( ! dzOptionString.compare("vz") ) dzOption = EgammaTrackSelector::vz;
    else if( ! dzOptionString.compare("bs") ) dzOption = EgammaTrackSelector::bs;
    else if( ! dzOptionString.compare("vtx") )dzOption = EgammaTrackSelector::vtx;
    else                                      dzOption = EgammaTrackSelector::dz;

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

    reco::isodeposit::Direction candDir;
    double dzCut=0;

    reco::TrackBase::Point beamPoint(0,0, 0);
    if (theBeamlineOption.compare("BeamSpotFromEvent") == 0){
        //pick beamSpot
        reco::BeamSpot beamSpot;
        edm::Handle<reco::BeamSpot> beamSpotH;

        event.getByToken(theBeamSpotToken,beamSpotH);

        if (beamSpotH.isValid()){
            beamPoint = beamSpotH->position();
        }
    }

    Handle<View<Track> > tracksH;
    event.getByToken(theTrackCollectionToken, tracksH);

    if( candTk.isElectron() ){
        const reco::GsfElectron* elec = dynamic_cast<const reco::GsfElectron*>(&candTk);
        candDir = reco::isodeposit::Direction(elec->gsfTrack()->eta(), elec->gsfTrack()->phi());
    } else {
        candDir = reco::isodeposit::Direction(candTk.eta(), candTk.phi());
    }

    IsoDeposit deposit(candDir );
    deposit.setVeto( veto(candDir) );
    deposit.addCandEnergy(candTk.et());

    View<Track>::const_iterator itrTr = tracksH->begin();
    View<Track>::const_iterator trEnd = tracksH->end();
    for (itrTr = tracksH->begin();itrTr != trEnd; ++itrTr) {

        if(candDir.deltaR( reco::isodeposit::Direction(itrTr->eta(),itrTr->phi()) ) > theDR_Max )
            continue;

        if(itrTr->normalizedChi2() > theChi2Ndof_Max)
            continue;

        if(itrTr->pt() < thePt_Min)
            continue;

        if(theChi2Prob_Min > 0 && ChiSquaredProbability(itrTr->chi2(), itrTr->ndof()) < theChi2Prob_Min )
            continue;

        if(theNHits_Min > 0 && itrTr->numberOfValidHits() < theNHits_Min)
            continue;

        if( candTk.isElectron() ){
            const reco::GsfElectron* elec = dynamic_cast<const reco::GsfElectron*>(&candTk);
            switch(dzOption) {
                case EgammaTrackSelector::dz : dzCut = elec->gsfTrack()->dz() - itrTr->dz() ; break;
                case EgammaTrackSelector::vz : dzCut = elec->gsfTrack()->vz() - itrTr->vz() ; break;
                case EgammaTrackSelector::bs : dzCut = elec->gsfTrack()->dz(beamPoint) - itrTr->dz(beamPoint) ; break;
                case EgammaTrackSelector::vtx: dzCut = itrTr->dz(elec->gsfTrack()->vertex()) ; break;
                default: dzCut = elec->gsfTrack()->vz() - itrTr->vz() ; break;
            }
        } else {
            switch(dzOption) {
                case EgammaTrackSelector::dz : dzCut = (*itrTr).dz() - candTk.vertex().z() ; break;
                case EgammaTrackSelector::vz : dzCut = (*itrTr).vz() - candTk.vertex().z() ; break;
                case EgammaTrackSelector::bs : dzCut = (*itrTr).dz(beamPoint) - candTk.vertex().z() ; break;
                case EgammaTrackSelector::vtx: dzCut = (*itrTr).dz(candTk.vertex()); break;
                default : dzCut = (*itrTr).vz() - candTk.vertex().z(); break;
            }
        }

        if(fabs(dzCut) > theDiff_z)
            continue;

        if(fabs(itrTr->dxy(beamPoint) ) > theDiff_r)
            continue;

        deposit.addDeposit(reco::isodeposit::Direction(itrTr->eta(), itrTr->phi()), itrTr->pt());

    }

    return deposit;
}
