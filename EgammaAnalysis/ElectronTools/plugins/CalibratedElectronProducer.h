// This file is imported from 

#ifndef CalibratedElectronProducer_h
#define CalibratedElectronProducer_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "EgammaAnalysis/ElectronTools/interface/SimpleElectron.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEPcombinator.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEnergyCalibrator.h"
#include "EgammaAnalysis/ElectronTools/interface/EpCombinationTool.h"


class CalibratedElectronProducer: public edm::EDProducer 
{
    public:
        explicit CalibratedElectronProducer( const edm::ParameterSet & ) ;
        virtual ~CalibratedElectronProducer();
        virtual void produce( edm::Event &, const edm::EventSetup & ) ;
    
    private:
        edm::InputTag inputElectrons_ ;
        edm::InputTag nameEnergyReg_;
        edm::InputTag nameEnergyErrorReg_;
        edm::InputTag recHitCollectionEB_ ;
        edm::InputTag recHitCollectionEE_ ;
        
        std::string nameNewEnergyReg_ ;
        std::string nameNewEnergyErrorReg_;
        
        std::string dataset ;
        bool isAOD ;
        bool isMC ;
        bool updateEnergyError ;
        int correctionsType ;
        bool applyLinearityCorrection;
        int combinationType ;
        bool verbose ;
        bool synchronization ;
        double lumiRatio;
        
        const CaloTopology * ecalTopology_;
        const CaloGeometry * caloGeometry_;
        bool geomInitialized_;
        std::string newElectronName_;
        std::string combinationRegressionInputPath;
        std::string scaleCorrectionsInputPath;
        std::string linCorrectionsInputPath;
        
        ElectronEnergyCalibrator *theEnCorrector;
        EpCombinationTool *myEpCombinationTool;
        ElectronEPcombinator *myCombinator;
};

#endif
