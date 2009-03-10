//*****************************************************************************
// File:      EgammaTowerExtractor.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************
//C++ includes
#include <vector>
#include <functional>
#include <math.h>

//ROOT includes
#include <Math/VectorUtil.h>


//CMSSW includes
#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaTowerExtractor.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"

using namespace ROOT::Math::VectorUtil ;

using namespace egammaisolation;
using namespace reco::isodeposit;

EgammaTowerExtractor::~EgammaTowerExtractor(){}

reco::IsoDeposit EgammaTowerExtractor::deposit(const edm::Event & iEvent, 
        const edm::EventSetup & iSetup, const reco::Candidate &emObject ) const {

    edm::Handle<CaloTowerCollection> towercollectionH;
    iEvent.getByLabel(caloTowerTag_, towercollectionH);
 
    //Take the SC position
    reco::SuperClusterRef sc = emObject.get<reco::SuperClusterRef>();
    math::XYZPoint caloPosition = sc->position();

    Direction candDir(caloPosition.eta(), caloPosition.phi());
    reco::IsoDeposit deposit( candDir );
    deposit.setVeto( reco::IsoDeposit::Veto(candDir, intRadius_) ); 
    deposit.addCandEnergy(sc->energy()*sin(2*atan(exp(-sc->eta()))));

    //loop over tracks
    for(CaloTowerCollection::const_iterator trItr = towercollectionH->begin(), trEnd = towercollectionH->end(); trItr != trEnd; ++trItr){
        double depEt  = trItr->hadEt();
        if ( depEt < etLow_ )  continue ;  

        Direction towerDir( trItr->eta(), trItr->phi() );
        double dR2 = candDir.deltaR2(towerDir);

        if (dR2 < extRadius2_) {
            deposit.addDeposit( towerDir, depEt);
        }

    }//end loop over tracks

    return deposit;
}

