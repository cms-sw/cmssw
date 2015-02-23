#include "RecoLocalCalo/EcalRecProducers/plugins/EcalUncalibRecHitWorkerRatio.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/EcalSampleMaskRcd.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

EcalUncalibRecHitWorkerRatio::EcalUncalibRecHitWorkerRatio(const edm::ParameterSet&ps, edm::ConsumesCollector& c) :
  EcalUncalibRecHitWorkerBaseClass(ps,c)
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
  
  // which of the samples need be used      
  es.get<EcalSampleMaskRcd>().get(sampleMaskHand_);
  
  es.get<EcalGainRatiosRcd>().get(gains);
  es.get<EcalPedestalsRcd>().get(peds);

}


bool
EcalUncalibRecHitWorkerRatio::run( const edm::Event & evt,
                const EcalDigiCollection::const_iterator & itdg,
                EcalUncalibratedRecHitCollection & result )
{
        DetId detid(itdg->id());

	const EcalSampleMask *sampleMask_ = sampleMaskHand_.product();

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
	    uncalibMaker_endcap_.makeRecHit(*itdg,*sampleMask_,pedVec,pedRMSVec,
					    gainRatios,EEtimeFitParameters_,
					    EEamplitudeFitParameters_,
					    EEtimeFitLimits_);//GF pass mask here
          
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
	    uncalibMaker_barrel_.makeRecHit(*itdg,*sampleMask_,pedVec,pedRMSVec,
					    gainRatios,EBtimeFitParameters_,
					    EBamplitudeFitParameters_,
					    EBtimeFitLimits_);//GF pass mask here
          
          
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

edm::ParameterSetDescription
EcalUncalibRecHitWorkerRatio::getAlgoDescription() {

  edm::ParameterSetDescription psd;
  std::vector<double> dSet1 = {-2.390548,3.553628,-17.62341,67.67538,-133.213,140.7432,-75.41106,16.20277};
  std::vector<double> dSet2 =  {-2.015452, 3.130702, -12.3473, 41.88921, -82.83944, 91.01147, -50.35761, 11.05621};
  
  psd.addNode(edm::ParameterDescription<double>("EEtimeFitLimits_Upper", 1.4, true) and
	       edm::ParameterDescription<double>("EEtimeConstantTerm", 0.18, true) and
	       edm::ParameterDescription<double>("EBtimeFitLimits_Lower", 0.2, true) and
	       edm::ParameterDescription<double>("EBtimeConstantTerm", 0.26, true) and
	       edm::ParameterDescription<double>("EEtimeFitLimits_Lower", 0.2, true) and
	       edm::ParameterDescription<std::vector<double> >("EEtimeFitParameters", dSet1, true) and
	       edm::ParameterDescription<std::vector<double>>("EEamplitudeFitParameters", {1.89, 1.4}, true) and
	       edm::ParameterDescription<double>("EBtimeFitLimits_Upper", 1.4, true) and
	       edm::ParameterDescription<std::vector<double>>("EBamplitudeFitParameters", {1.138,1.652}, true) and
	       edm::ParameterDescription<std::vector<double>>("EBtimeFitParameters", dSet2, true) );

  return psd;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalUncalibRecHitWorkerFactory, EcalUncalibRecHitWorkerRatio, "EcalUncalibRecHitWorkerRatio" );
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitFillDescriptionWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalUncalibRecHitFillDescriptionWorkerFactory, EcalUncalibRecHitWorkerRatio, "EcalUncalibRecHitWorkerRatio");
