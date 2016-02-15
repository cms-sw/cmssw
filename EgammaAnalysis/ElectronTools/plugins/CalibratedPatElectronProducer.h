// This file is imported from
// http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/UserCode/Mangano/WWAnalysis/AnalysisStep/interface/CalibratedPatElectronProducer.h?revision=1.1&view=markup

#ifndef CalibratedPatElectronProducer_h
#define CalibratedPatElectronProducer_h

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "EgammaAnalysis/ElectronTools/interface/SimpleElectron.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEPcombinator.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEnergyCalibrator.h"
#include "EgammaAnalysis/ElectronTools/interface/EpCombinationTool.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class CalibratedPatElectronProducer: public edm::EDProducer
{
    public:
        explicit CalibratedPatElectronProducer( const edm::ParameterSet & ) ;
        virtual ~CalibratedPatElectronProducer();
        virtual void produce( edm::Event &, const edm::EventSetup & ) ;

    private:
        edm::EDGetTokenT<edm::View<reco::Candidate> > inputPatElectronsToken ;
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
        std::string combinationRegressionInputPath;
        std::string scaleCorrectionsInputPath;
        std::string linCorrectionsInputPath;
	bool applyExtraHighEnergyProtection;

        ElectronEnergyCalibrator *theEnCorrector;
        EpCombinationTool *myEpCombinationTool;
        ElectronEPcombinator *myCombinator;
};

#endif
