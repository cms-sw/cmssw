//*****************************************************************************
// File:      EgammaRecHitExtractor.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer, hacked by Sam Harper (ie the ugly stuff is mine)
// Institute: IIHE-VUB, RAL
//=============================================================================
//*****************************************************************************
//C++ includes
#include <vector>
#include <functional>

//ROOT includes
#include <Math/VectorUtil.h>

//CMSSW includes
#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaRecHitExtractor.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace egammaisolation;
using namespace reco::isodeposit;

EgammaRecHitExtractor::EgammaRecHitExtractor(const edm::ParameterSet& par) : 
    etMin_(par.getParameter<double>("etMin")),
    energyMin_(par.getParameter<double>("energyMin")),
    minCandEt_(par.getParameter<double>("minCandEt")),
    extRadius_(par.getParameter<double>("extRadius")),
    intRadius_(par.getParameter<double>("intRadius")),
    intStrip_(par.getParameter<double>("intStrip")),
    barrelRecHitsTag_(par.getParameter<edm::InputTag>("barrelRecHits")), 
    endcapRecHitsTag_(par.getParameter<edm::InputTag>("endcapRecHits")),
    barrelEcalHitsTag_(par.getParameter<edm::InputTag>("barrelEcalHits")), 
    endcapEcalHitsTag_(par.getParameter<edm::InputTag>("endcapEcalHits")),
    fakeNegativeDeposit_(par.getParameter<bool>("subtractSuperClusterEnergy")),
    tryBoth_(par.getParameter<bool>("tryBoth")),
    sameTag_(false)
{ 
    if ((intRadius_ != 0.0) && (fakeNegativeDeposit_)) {
        throw cms::Exception("Configuration Error") << "EgammaRecHitExtractor: " << 
            "If you use 'subtractSuperClusterEnergy', you *must* set 'intRadius' to ZERO; it does not make sense, otherwise.";
    }
    std::string isoVariable = par.getParameter<std::string>("isolationVariable");
    if (isoVariable == "et") {
        useEt_ = true;
    } else if (isoVariable == "energy") {
        useEt_ = false;
    } else {
        throw cms::Exception("Configuration Error") << "EgammaRecHitExtractor: isolationVariable '" << isoVariable << "' not known. " 
            << " Supported values are 'et', 'energy'. ";
    }
    std::string detector = par.getParameter<std::string>("detector");
    if (detector == "Ecal") {
        detector_ = DetId::Ecal;
    } else if (detector == "Hcal") {
        detector_ = DetId::Hcal;
    } else {
        throw cms::Exception("Configuration Error") << "EgammaRecHitExtractor: detector '" << detector << "' not known. " 
            << " Supported values are 'Ecal', 'Hcal'. ";
    }

    if (endcapRecHitsTag_.encode() ==  barrelRecHitsTag_.encode()) {
        sameTag_ = true;
        if (tryBoth_) {
            edm::LogWarning("EgammaRecHitExtractor") << "If you have configured 'barrelRecHits' == 'endcapRecHits', so I'm switching 'tryBoth' to FALSE.";
            tryBoth_ = false;
        }
    }

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

EgammaRecHitExtractor::~EgammaRecHitExtractor() { }

reco::IsoDeposit EgammaRecHitExtractor::deposit(const edm::Event & iEvent, 
        const edm::EventSetup & iSetup, const reco::Candidate &emObject ) const {
    edm::ESHandle<CaloGeometry> pG;
    iSetup.get<CaloGeometryRecord>().get(pG);
    const CaloGeometry* caloGeom = pG.product(); 

    static std::string metname = "EgammaIsolationAlgos|EgammaRecHitExtractor";

    std::auto_ptr<const CaloRecHitMetaCollectionV> barrelRecHits(0), endcapRecHits(0);

    //Get barrel ECAL RecHits for Iso checking
    edm::Handle<EcalRecHitCollection> barrelEcalRecHitsH;
    iEvent.getByLabel(barrelEcalHitsTag_, barrelEcalRecHitsH);

    //Get endcap ECAL RecHits for Iso checking
    edm::Handle<EcalRecHitCollection> endcapEcalRecHitsH;
    iEvent.getByLabel(endcapEcalHitsTag_, endcapEcalRecHitsH);

    if (detector_ == DetId::Ecal) {
        barrelRecHits = std::auto_ptr<const CaloRecHitMetaCollectionV>(new EcalRecHitMetaCollection(*barrelEcalRecHitsH));
        if (!sameTag_) {
            endcapRecHits = std::auto_ptr<const CaloRecHitMetaCollectionV>(new EcalRecHitMetaCollection(*endcapEcalRecHitsH));
        }
    } else if (detector_ == DetId::Hcal) {
        edm::Handle<HBHERecHitCollection> barrelHcalRecHitsH, endcapHcalRecHitsH;
        iEvent.getByLabel(barrelRecHitsTag_, barrelHcalRecHitsH);
        barrelRecHits = std::auto_ptr<const CaloRecHitMetaCollectionV>(new HBHERecHitMetaCollection(*barrelHcalRecHitsH));
        if (!sameTag_) {
            iEvent.getByLabel(endcapRecHitsTag_, endcapHcalRecHitsH);
            endcapRecHits = std::auto_ptr<const CaloRecHitMetaCollectionV>(new HBHERecHitMetaCollection(*endcapHcalRecHitsH));
        }
    }

    reco::SuperClusterRef sc = emObject.get<reco::SuperClusterRef>();
    math::XYZPoint caloPosition = sc->position();
    GlobalPoint point(caloPosition.x(), caloPosition.y() , caloPosition.z());

    Direction candDir(caloPosition.eta(), caloPosition.phi());
    reco::IsoDeposit deposit( candDir );
    deposit.setVeto( reco::IsoDeposit::Veto(candDir, intRadius_) ); 
    double sinTheta = sin(2*atan(exp(-sc->eta())));
    deposit.addCandEnergy(sc->energy() * (useEt_ ? sinTheta : 1.0)) ;

    //avoid slow preshower
    int bid,eid;
    if(detector_==DetId::Ecal){
        bid=EcalBarrel;
        eid=EcalEndcap;
    }else{
        bid=0;eid=0;
    }

    double fakeEnergy = -sc->rawEnergy();
    if (fakeNegativeDeposit_) {
        deposit.addDeposit(candDir, fakeEnergy * (useEt_ ?  sinTheta : 1.0)); // not exactly clean...
    }

    std::auto_ptr<CaloRecHitMetaCollectionV> ecalRecHits(0); 
    double extRadius, innRadius, etaStrip, minEtRecHit, isolEtCut;
    if( abs(sc->eta()) < 1.5 ) {
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
    if ( sc->energy()*sinTheta < minCandEt_ || candIso.getEtSum(&emObject) > isolEtCut ) {
        deposit.addDeposit( Direction(caloPosition.eta(), caloPosition.phi()+0.15), 10000 );
        deposit.addDeposit( Direction(caloPosition.eta(), caloPosition.phi()+0.25), 100000 );
    } else {
        CaloDualConeSelector doubleConeSelBarrel(intRadius_ ,extRadius_, caloGeom, detector_,bid);
        CaloDualConeSelector doubleConeSelEndcap(intRadius_ ,extRadius_, caloGeom, detector_,eid);

        // 3 possible options
        bool inBarrel = sameTag_ || ( abs(sc->eta()) < 1.5 );
        if (inBarrel || tryBoth_) {
            collect(deposit, point, doubleConeSelBarrel, caloGeom, *barrelRecHits);
        } 
        if ((!inBarrel) || tryBoth_) {
            collect(deposit, point, doubleConeSelEndcap, caloGeom, *endcapRecHits);
        }
    }
    return deposit;
}

void EgammaRecHitExtractor::collect(reco::IsoDeposit &deposit, 
        const GlobalPoint &caloPosition, CaloDualConeSelector &cone, 
        const CaloGeometry* caloGeom,
        const CaloRecHitMetaCollectionV &hits) const 
{
    std::auto_ptr<CaloRecHitMetaCollectionV> chosen = cone.select(caloPosition, hits);
    for (CaloRecHitMetaCollectionV::const_iterator i = chosen->begin(), end = chosen->end() ; i != end;  ++i)  {
        const  GlobalPoint & position = caloGeom->getPosition(i->detid());
        double eta = position.eta();
        double energy = i->energy();
        double et = energy*sin(2*atan(exp(-eta)));
        if ( et > etMin_ && energy > energyMin_ && fabs(eta-caloPosition.eta()) > intStrip_ ){
            deposit.addDeposit( Direction(eta, position.phi()), (useEt_ ? et : energy) );
        }
    }
} 


