#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerFixedAlphaBetaFit_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerFixedAlphaBetaFit_hh

#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerBaseClass.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitFixedAlphaBetaAlgo.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"

namespace edm {
        class Event;
        class EventSetup;
        class ParameterSet;
        class ParameterSetDescription;
}

class EcalUncalibRecHitWorkerFixedAlphaBetaFit : public EcalUncalibRecHitWorkerBaseClass {

        public:
                EcalUncalibRecHitWorkerFixedAlphaBetaFit(const edm::ParameterSet& ps, edm::ConsumesCollector& ); 
		EcalUncalibRecHitWorkerFixedAlphaBetaFit() {};
                virtual ~EcalUncalibRecHitWorkerFixedAlphaBetaFit() {};

                void set(const edm::EventSetup& es);
                bool run(const edm::Event& evt, const EcalDigiCollection::const_iterator & digi, EcalUncalibratedRecHitCollection & result);
		
		edm::ParameterSetDescription getAlgoDescription();
        private:

                double AmplThrEB_;
                double AmplThrEE_;

                EcalUncalibRecHitFixedAlphaBetaAlgo<EBDataFrame> algoEB_;
                EcalUncalibRecHitFixedAlphaBetaAlgo<EEDataFrame> algoEE_;

                double alphaEB_;
                double betaEB_;
                double alphaEE_;
                double betaEE_;
                std::vector<std::vector<std::pair<double,double> > > alphaBetaValues_; // List of alpha and Beta values [SM#][CRY#](alpha, beta)
                bool useAlphaBetaArray_;
                std::string alphabetaFilename_;

                bool setAlphaBeta(); // Sets the alphaBetaValues_ vectors by the values provided in alphabetaFilename_

                edm::ESHandle<EcalGainRatios> pRatio;
                edm::ESHandle<EcalPedestals> pedHandle;
};
#endif
