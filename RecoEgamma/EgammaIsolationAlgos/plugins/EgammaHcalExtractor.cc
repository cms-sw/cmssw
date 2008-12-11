//*****************************************************************************
// File:      EgammaHcalExtractor.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************
//C++ includes
#include <vector>
#include <functional>

//ROOT includes
#include <Math/VectorUtil.h>

//CMSSW includes
#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaHcalExtractor.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"
#include "RecoCaloTools/Selectors/interface/CaloDualConeSelector.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

using namespace std;

using namespace egammaisolation;
using namespace reco::isodeposit;

EgammaHcalExtractor::EgammaHcalExtractor ( const edm::ParameterSet& par ) :
    minCandEt_(par.getParameter<double>("minCandEt")),
    extRadius_(par.getParameter<double>("extRadius")),
    intRadius_(par.getParameter<double>("intRadius")),
    etLow_(par.getParameter<double>("etMin")),
    barrelEcalHitsTag_(par.getParameter<edm::InputTag>("barrelEcalHits")),
    endcapEcalHitsTag_(par.getParameter<edm::InputTag>("endcapEcalHits")),
    hcalRecHitProducer_(par.getParameter<edm::InputTag>("hcalRecHits")) { 

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

EgammaHcalExtractor::~EgammaHcalExtractor(){}

reco::IsoDeposit EgammaHcalExtractor::deposit(const edm::Event & iEvent, 
        const edm::EventSetup & iSetup, const reco::Candidate &emObject ) const {

    //Get MetaRecHit collection
    edm::Handle<HBHERecHitCollection> hcalRecHitHandle;
    iEvent.getByLabel(hcalRecHitProducer_, hcalRecHitHandle);
    HBHERecHitMetaCollection mhbhe =  HBHERecHitMetaCollection(*hcalRecHitHandle); 

    //Get Calo Geometry
    edm::ESHandle<CaloGeometry> pG;
    iSetup.get<CaloGeometryRecord>().get(pG);
    const CaloGeometry* caloGeom = pG.product();
    CaloDualConeSelector coneSel(intRadius_, extRadius_, caloGeom, DetId::Hcal);

    //Get barrel ECAL RecHits for Iso checking
    edm::Handle<EcalRecHitCollection> barrelEcalRecHitsH;
    iEvent.getByLabel(barrelEcalHitsTag_, barrelEcalRecHitsH);

    //Get endcap ECAL RecHits for Iso checking
    edm::Handle<EcalRecHitCollection> endcapEcalRecHitsH;
    iEvent.getByLabel(endcapEcalHitsTag_, endcapEcalRecHitsH);

    //Take the SC position
    reco::SuperClusterRef sc = emObject.get<reco::SuperClusterRef>();
    math::XYZPoint caloPosition = sc->position();
    GlobalPoint point(caloPosition.x(), caloPosition.y() , caloPosition.z());
    // needed: coneSel.select(eta,phi,hits) is not the same!

    Direction candDir(caloPosition.eta(), caloPosition.phi());
    reco::IsoDeposit deposit( candDir );
    deposit.setVeto( reco::IsoDeposit::Veto(candDir, intRadius_) ); 
    double sinTheta = sin(2*atan(exp(-sc->eta())));
    deposit.addCandEnergy(sc->energy()*sinTheta);

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
        //Compute the HCAL energy behind ECAL
        std::auto_ptr<CaloRecHitMetaCollectionV> chosen = coneSel.select(point, mhbhe);
        for (CaloRecHitMetaCollectionV::const_iterator i = chosen->begin (), ed = chosen->end() ; 
                i!= ed; ++i) {
            const  GlobalPoint & hcalHit_position = caloGeom->getPosition(i->detid());
            double hcalHit_eta = hcalHit_position.eta();
            double hcalHit_Et = i->energy()*sin(2*atan(exp(-hcalHit_eta)));
            if ( hcalHit_Et > etLow_) {
                deposit.addDeposit( Direction(hcalHit_eta, hcalHit_position.phi()), hcalHit_Et);
            }
        }
    }

    return deposit;
}
