#include "RecoLocalCalo/EcalRecProducers/plugins/EcalUncalibRecHitWorkerRatio.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

EcalUncalibRecHitWorkerRatio::EcalUncalibRecHitWorkerRatio(const edm::ParameterSet&ps) :
        EcalUncalibRecHitWorkerBaseClass(ps)
{
  EBtimeFitParameters_ = ps.getParameter<std::vector<double> >("EBtimeFitParameters"); 
  EEtimeFitParameters_ = ps.getParameter<std::vector<double> >("EEtimeFitParameters"); 
 
  EBamplitudeFitParameters_ = ps.getParameter<std::vector<double> >("EBamplitudeFitParameters"); 
  EEamplitudeFitParameters_ = ps.getParameter<std::vector<double> >("EEamplitudeFitParameters"); 
 
  EBtimeFitLimits_.first  = ps.getParameter<double>("EBtimeFitLimits_Lower"); 
  EBtimeFitLimits_.second = ps.getParameter<double>("EBtimeFitLimits_Upper"); 
 
  EEtimeFitLimits_.first  = ps.getParameter<double>("EEtimeFitLimits_Lower"); 
  EEtimeFitLimits_.second = ps.getParameter<double>("EEtimeFitLimits_Upper"); 

  EBtimeConstantTerm_ = ps.getParameter<double>("EBtimeConstantTerm");
  EEtimeConstantTerm_ = ps.getParameter<double>("EEtimeConstantTerm");
}

void
EcalUncalibRecHitWorkerRatio::set(const edm::EventSetup& es)
{
        es.get<EcalGainRatiosRcd>().get(gains);
        es.get<EcalPedestalsRcd>().get(peds);

}


bool
EcalUncalibRecHitWorkerRatio::run( const edm::Event & evt,
                const EcalDigiCollection::const_iterator & itdg,
                EcalUncalibratedRecHitCollection & result )
{
        DetId detid(itdg->id());

        const EcalPedestals::Item * aped = 0;
        const EcalMGPAGainRatio * aGain = 0;

        if (detid.subdetId()==EcalEndcap) {
                unsigned int hashedIndex = EEDetId(detid).hashedIndex();
                aped  = &peds->endcap(hashedIndex);
                aGain = &gains->endcap(hashedIndex);
        } else {
                unsigned int hashedIndex = EBDetId(detid).hashedIndex();
                aped  = &peds->barrel(hashedIndex);
                aGain = &gains->barrel(hashedIndex);
        }

        pedVec[0] = aped->mean_x12;
        pedVec[1] = aped->mean_x6;
        pedVec[2] = aped->mean_x1;
        pedRMSVec[0] = aped->rms_x12;
        pedRMSVec[1] = aped->rms_x6;
        pedRMSVec[2] = aped->rms_x1;
        gainRatios[0] = 1.;
        gainRatios[1] = aGain->gain12Over6();
        gainRatios[2] = aGain->gain6Over1()*aGain->gain12Over6();

        float clockToNsConstant = 25.;
        EcalUncalibratedRecHit uncalibRecHit;

	if (detid.subdetId()==EcalEndcap) {

          uncalibRecHit = 
	    uncalibMaker_endcap_.makeRecHit(*itdg,pedVec,pedRMSVec,
					    gainRatios,EEtimeFitParameters_,
					    EEamplitudeFitParameters_,
					    EEtimeFitLimits_);
          
          EcalUncalibRecHitRatioMethodAlgo<EEDataFrame>::CalculatedRecHit crh =
	                          uncalibMaker_endcap_.getCalculatedRecHit();
          uncalibRecHit.setAmplitude( crh.amplitudeMax );
          uncalibRecHit.setJitter( crh.timeMax - 5 );
          uncalibRecHit.setJitterError( std::sqrt(pow(crh.timeError,2) + 
				        std::pow(EEtimeConstantTerm_,2)/
                                        std::pow(clockToNsConstant,2)) );

        } else {
 

	  bool gainSwitch = uncalibMaker_barrel_.fixMGPAslew(*itdg);

          uncalibRecHit= 
	    uncalibMaker_barrel_.makeRecHit(*itdg,pedVec,pedRMSVec,
					    gainRatios,EBtimeFitParameters_,
					    EBamplitudeFitParameters_,
					    EBtimeFitLimits_);
          
          
          EcalUncalibRecHitRatioMethodAlgo<EBDataFrame>::CalculatedRecHit crh= 
	    uncalibMaker_barrel_.getCalculatedRecHit();

          uncalibRecHit.setAmplitude( crh.amplitudeMax );
	  if(gainSwitch){
	    // introduce additional 1ns shift
	    uncalibRecHit.setJitter( crh.timeMax - 5 - 0.04 );
	  }else{
	    uncalibRecHit.setJitter( crh.timeMax - 5);
	  }
          uncalibRecHit.setJitterError( std::sqrt(pow(crh.timeError,2) + 
					std::pow(EBtimeConstantTerm_,2)/
                                        std::pow(clockToNsConstant,2)) );

        }
        result.push_back(uncalibRecHit);

        return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalUncalibRecHitWorkerFactory, EcalUncalibRecHitWorkerRatio, "EcalUncalibRecHitWorkerRatio" );
