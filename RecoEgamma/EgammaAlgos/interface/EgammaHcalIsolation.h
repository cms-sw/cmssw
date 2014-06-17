#ifndef EgammaIsolationAlgos_EgammaHcalIsolation_h
#define EgammaIsolationAlgos_EgammaHcalIsolation_h
//*****************************************************************************
// File:      EgammaHcalIsolation.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

//C++ includes
#include <vector>
#include <functional>

//CMSSW includes
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoCaloTools/Selectors/interface/CaloDualConeSelector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

//Sum helper functions
double scaleToE(const double& eta);
double scaleToEt(const double& eta);


class EgammaHcalIsolation {
    public:

        enum HcalDepth{AllDepths=0,Depth1=1,Depth2=2};

        //constructors
        EgammaHcalIsolation (
                double extRadius,
                double intRadius,
                double eLowB,
                double eLowE,
                double etLowB,
                double etLowE,
                edm::ESHandle<CaloGeometry> theCaloGeom ,
                const HBHERecHitCollection&  mhbhe
                );

        //destructor 
        ~EgammaHcalIsolation() ;

        //AllDepths
        double getHcalESum (const reco::Candidate *c)     const { return getHcalESum(c->get<reco::SuperClusterRef>().get()); } 
        double getHcalEtSum(const reco::Candidate *c)     const { return getHcalEtSum(c->get<reco::SuperClusterRef>().get()); } 
        double getHcalESum (const reco::SuperCluster *sc) const { return getHcalESum(sc->position()); } 
        double getHcalEtSum(const reco::SuperCluster *sc) const { return getHcalEtSum(sc->position()); } 
        double getHcalESum (const math::XYZPoint &p)      const { return getHcalESum(GlobalPoint(p.x(),p.y(),p.z())); } 
        double getHcalEtSum(const math::XYZPoint &p)      const { return getHcalEtSum(GlobalPoint(p.x(),p.y(),p.z())); } 
        double getHcalESum (const GlobalPoint &pclu)      const { return getHcalSum(pclu,AllDepths,&scaleToE); } 
        double getHcalEtSum(const GlobalPoint &pclu)      const { return getHcalSum(pclu,AllDepths,&scaleToEt); }

        //Depth1
        double getHcalESumDepth1 (const reco::Candidate *c)     const { return getHcalESumDepth1(c->get<reco::SuperClusterRef>().get()); } 
        double getHcalEtSumDepth1(const reco::Candidate *c)     const { return getHcalEtSumDepth1(c->get<reco::SuperClusterRef>().get()); } 
        double getHcalESumDepth1 (const reco::SuperCluster *sc) const { return getHcalESumDepth1(sc->position()); } 
        double getHcalEtSumDepth1(const reco::SuperCluster *sc) const { return getHcalEtSumDepth1(sc->position()); } 
        double getHcalESumDepth1 (const math::XYZPoint &p)      const { return getHcalESumDepth1(GlobalPoint(p.x(),p.y(),p.z())); } 
        double getHcalEtSumDepth1(const math::XYZPoint &p)      const { return getHcalEtSumDepth1(GlobalPoint(p.x(),p.y(),p.z())); } 
        double getHcalESumDepth1 (const GlobalPoint &pclu)      const { return getHcalSum(pclu,Depth1,&scaleToE); } 
        double getHcalEtSumDepth1(const GlobalPoint &pclu)      const { return getHcalSum(pclu,Depth1,&scaleToEt); }

        //Depth2
        double getHcalESumDepth2 (const reco::Candidate *c)     const { return getHcalESumDepth2(c->get<reco::SuperClusterRef>().get()); } 
        double getHcalEtSumDepth2(const reco::Candidate *c)     const { return getHcalEtSumDepth2(c->get<reco::SuperClusterRef>().get()); } 
        double getHcalESumDepth2 (const reco::SuperCluster *sc) const { return getHcalESumDepth2(sc->position()); } 
        double getHcalEtSumDepth2(const reco::SuperCluster *sc) const { return getHcalEtSumDepth2(sc->position()); } 
        double getHcalESumDepth2 (const math::XYZPoint &p)      const { return getHcalESumDepth2(GlobalPoint(p.x(),p.y(),p.z())); } 
        double getHcalEtSumDepth2(const math::XYZPoint &p)      const { return getHcalEtSumDepth2(GlobalPoint(p.x(),p.y(),p.z())); } 
        double getHcalESumDepth2 (const GlobalPoint &pclu)      const { return getHcalSum(pclu,Depth2,&scaleToE); } 
        double getHcalEtSumDepth2(const GlobalPoint &pclu)      const { return getHcalSum(pclu,Depth2,&scaleToEt); }


    private:

        
        bool isDepth2(const DetId&) const;
        double getHcalSum(const GlobalPoint&, const HcalDepth&, double(*)(const double&) ) const;

        double extRadius_ ;
        double intRadius_ ;
        double eLowB_ ;
        double eLowE_ ;
        double etLowB_ ;
        double etLowE_ ;


        edm::ESHandle<CaloGeometry>  theCaloGeom_ ;
        const HBHERecHitCollection&  mhbhe_ ;

        CaloDualConeSelector<HBHERecHit>* doubleConeSel_;

};

#endif
