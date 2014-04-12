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

EgammaHcalExtractor::EgammaHcalExtractor ( const edm::ParameterSet& par, edm::ConsumesCollector & iC ) :
    extRadius_(par.getParameter<double>("extRadius")),
    intRadius_(par.getParameter<double>("intRadius")),
    etLow_(par.getParameter<double>("etMin")),
    hcalRecHitProducerToken_(iC.consumes<HBHERecHitCollection>(par.getParameter<edm::InputTag>("hcalRecHits"))) {
}

EgammaHcalExtractor::~EgammaHcalExtractor(){}

reco::IsoDeposit EgammaHcalExtractor::deposit(const edm::Event & iEvent,
        const edm::EventSetup & iSetup, const reco::Candidate &emObject ) const {

    //Get MetaRecHit collection
    edm::Handle<HBHERecHitCollection> hcalRecHitHandle;
    iEvent.getByToken(hcalRecHitProducerToken_, hcalRecHitHandle);

    //Get Calo Geometry
    edm::ESHandle<CaloGeometry> pG;
    iSetup.get<CaloGeometryRecord>().get(pG);
    const CaloGeometry* caloGeom = pG.product();
    CaloDualConeSelector<HBHERecHit> coneSel(intRadius_, extRadius_, caloGeom, DetId::Hcal);

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

    //Compute the HCAL energy behind ECAL
	coneSel.selectCallback(point, *hcalRecHitHandle, [&](const HBHERecHit& i) {
        const  GlobalPoint & hcalHit_position = caloGeom->getPosition(i.detid());
        double hcalHit_eta = hcalHit_position.eta();
        double hcalHit_Et = i.energy()*sin(2*atan(exp(-hcalHit_eta)));
        if ( hcalHit_Et > etLow_) {
            deposit.addDeposit( Direction(hcalHit_eta, hcalHit_position.phi()), hcalHit_Et);
        }
	});

    return deposit;
}
