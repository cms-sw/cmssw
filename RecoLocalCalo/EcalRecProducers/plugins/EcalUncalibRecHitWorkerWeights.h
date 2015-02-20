#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitRecWorkerWeights_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitRecWorkerWeights_hh

/** \class EcalUncalibRecHitRecWeightsAlgo
  *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using a weights method
  *
  *  \author R. Bruneliere - A. Zabi
  */

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecWeightsAlgo.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"

#include "SimCalorimetry/EcalSimAlgos/interface/EBShape.h"   
#include "SimCalorimetry/EcalSimAlgos/interface/EEShape.h"



namespace edm {
        class Event;
        class EventSetup;
        class ParameterSet;
	class ParameterSetDescription;
}

class EcalUncalibRecHitWorkerWeights : public EcalUncalibRecHitWorkerBaseClass {

        public:
                EcalUncalibRecHitWorkerWeights(const edm::ParameterSet&, edm::ConsumesCollector& c);
		EcalUncalibRecHitWorkerWeights() {};
                virtual ~EcalUncalibRecHitWorkerWeights() {};

                void set(const edm::EventSetup& es);
                bool run(const edm::Event& evt, const EcalDigiCollection::const_iterator & digi, EcalUncalibratedRecHitCollection & result);

		edm::ParameterSetDescription getAlgoDescription();

        protected:

                edm::ESHandle<EcalPedestals> peds;
                edm::ESHandle<EcalGainRatios>  gains;
                edm::ESHandle<EcalWeightXtalGroups>  grps;
                edm::ESHandle<EcalTBWeights> wgts;

                double pedVec[3];
                double pedRMSVec[3];
                double gainRatios[3];

                const EcalWeightSet::EcalWeightMatrix* weights[2];
                const EcalWeightSet::EcalChi2WeightMatrix* chi2mat[2];

                EcalUncalibRecHitRecWeightsAlgo<EBDataFrame> uncalibMaker_barrel_;
                EcalUncalibRecHitRecWeightsAlgo<EEDataFrame> uncalibMaker_endcap_;

 		const EEShape testbeamEEShape; // used in the chi2
                const EBShape testbeamEBShape; // can be replaced by simple shape arrays of floats in the future (kostas)

};

#endif
