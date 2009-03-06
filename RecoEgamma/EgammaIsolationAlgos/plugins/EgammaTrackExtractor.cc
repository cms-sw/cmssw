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
    minCandEt_(par.getParameter<double>("minCandEt")),
    theDiff_r(par.getParameter<double>("Diff_r")),
    theDiff_z(par.getParameter<double>("Diff_z")),
    theDR_Max(par.getParameter<double>("DR_Max")),
    theDR_Veto(par.getParameter<double>("DR_Veto")),
    theBeamlineOption(par.getParameter<string>("BeamlineOption")),
    barrelEcalHitsTag_(par.getParameter<edm::InputTag>("barrelEcalHits")),
    endcapEcalHitsTag_(par.getParameter<edm::InputTag>("endcapEcalHits")),
    theBeamSpotLabel(par.getParameter<edm::InputTag>("BeamSpotLabel")),
    theNHits_Min(par.getParameter<uint>("NHits_Min")),
    theChi2Ndof_Max(par.getParameter<double>("Chi2Ndof_Max")),
    theChi2Prob_Min(par.getParameter<double>("Chi2Prob_Min")),
    thePt_Min(par.getParameter<double>("Pt_Min"))
{
    paramForIsolBarrel_.push_back(par.getParameter<double>("checkIsoExtRBarrel"));
    paramForIsolBarrel_.push_back(par.getParameter<double>("checkIsoInnRBarrel"));
    paramForIsolBarrel_.push_back(par.getParameter<double>("checkIsoEtaStripBarrel"));
    paramForIsolBarrel_.push_back(par.getParameter<double>("checkIsoEtRecHitBarrel"));
    paramForIsolBarrel_.push_back(par.getParameter<double>("checkIsoEtCutBarrel"));

    paramForIsolEndcap_.push_back(par.getParameter<double>("checkIsoExtREndcap"));
    paramForIsolEndcap_.push_back(par.getParameter<double>("checkIsoInnREndcap"));
    paramForIsolEndcap_.push_back(par.getParameter<double>("checkIsoEtaStripEndcap"));
    paramForIsolEndcap_.push_back(par.getParameter<double>("checkIsoEtRecHitEndcap"));
    paramForIsolEndcap_.push_back(par.getParameter<double>("checkIsoEtCutEndcap"));
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

    reco::isodeposit::Direction muonDir(candTk.eta(), candTk.phi());

    IsoDeposit deposit(muonDir );
    deposit.setVeto( veto(muonDir) );
    deposit.addCandEnergy(candTk.et());

    Handle<View<Track> > tracksH;
    event.getByLabel(theTrackCollectionTag, tracksH);
    //  const TrackCollection tracks = *(tracksH.product());
    LogTrace(metname)<<"***** TRACK COLLECTION SIZE: "<<tracksH->size();

    double vtx_z = candTk.vertex().z();
    LogTrace(metname)<<"***** Muon vz: "<<vtx_z;
    reco::TrackBase::Point beamPoint(0,0, 0);

    if (theBeamlineOption.compare("BeamSpotFromEvent") == 0){
        //pick beamSpot
        reco::BeamSpot beamSpot;
        edm::Handle<reco::BeamSpot> beamSpotH;

        event.getByLabel(theBeamSpotLabel,beamSpotH);

        if (beamSpotH.isValid()){
            beamPoint = beamSpotH->position();
            LogTrace(metname)<<"Extracted beam point at "<<beamPoint<<std::endl;
        }
    }

    LogTrace(metname)<<"Using beam point at "<<beamPoint<<std::endl;

    //Get barrel ECAL RecHits for Iso checking
    edm::Handle<EcalRecHitCollection> barrelEcalRecHitsH;
    event.getByLabel(barrelEcalHitsTag_, barrelEcalRecHitsH);

    //Get endcap ECAL RecHits for Iso checking
    edm::Handle<EcalRecHitCollection> endcapEcalRecHitsH;
    event.getByLabel(endcapEcalHitsTag_, endcapEcalRecHitsH);

    edm::ESHandle<CaloGeometry> pG;
    eventSetup.get<CaloGeometryRecord>().get(pG);

    std::auto_ptr<CaloRecHitMetaCollectionV> ecalRecHits(0);
    double extRadius, innRadius, etaStrip, minEtRecHit, isolEtCut;
    if( abs(candTk.eta()) < 1.5 ) {
        extRadius   = paramForIsolBarrel_[0];
        innRadius   = paramForIsolBarrel_[1];
        etaStrip    = paramForIsolBarrel_[2];
        minEtRecHit = paramForIsolBarrel_[3];
        isolEtCut   = paramForIsolBarrel_[4];
        ecalRecHits = std::auto_ptr<CaloRecHitMetaCollectionV>(new EcalRecHitMetaCollection(*barrelEcalRecHitsH));
    } else {
        extRadius   = paramForIsolEndcap_[0];
        innRadius   = paramForIsolEndcap_[1];
        etaStrip    = paramForIsolEndcap_[2];
        minEtRecHit = paramForIsolEndcap_[3];
        isolEtCut   = paramForIsolEndcap_[4];
        ecalRecHits = std::auto_ptr<CaloRecHitMetaCollectionV>(new EcalRecHitMetaCollection(*endcapEcalRecHitsH));
    }

    EgammaRecHitIsolation candIso(extRadius,innRadius,etaStrip,minEtRecHit,pG,&(*ecalRecHits),DetId::Ecal);
    if ( candTk.et() < minCandEt_ || candIso.getEtSum(&candTk) > isolEtCut ) {
        deposit.addDeposit( Direction(candTk.eta(), candTk.phi()+0.15), 10000 );
        deposit.addDeposit( Direction(candTk.eta(), candTk.phi()+0.25), 100000 );
    } else {
        EgammaTrackSelector::Parameters pars(EgammaTrackSelector::Range(vtx_z-theDiff_z, vtx_z+theDiff_z),
                theDiff_r, muonDir, theDR_Max, beamPoint);

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
            LogTrace(metname) << "This track has: pt= " << tk->pt() << ", eta= " 
                << tk->eta() <<", phi= "<<tk->phi();
            reco::isodeposit::Direction dirTrk(tk->eta(), tk->phi());
            deposit.addDeposit(dirTrk, tk->pt());
        }

    }
    return deposit;
}
