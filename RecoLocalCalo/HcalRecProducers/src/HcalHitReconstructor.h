#ifndef HCALHITRECONSTRUCTOR_H 
#define HCALHITRECONSTRUCTOR_H 1

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSimpleRecAlgo.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHFStatusBitFromRecHits.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHFStatusBitFromDigis.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParams.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParam.h" 
#include "RecoLocalCalo/HcalRecAlgos/interface/HBHEStatusBitSetter.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalTimingCorrector.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HBHETimeProfileStatusBitSetter.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HBHETimingShapedFlag.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HBHEPulseShapeFlag.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalADCSaturationFlag.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HFTimingTrustFlag.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHF_S9S1algorithm.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHF_PETalgorithm.h"

    /** \class HcalHitReconstructor
	
    $Date: 2011/05/10 16:06:10 $
    $Revision: 1.13.10.1 $
    \author J. Temple & E. Yazgan
    ** Based on HcalSimpleReconstructor.h by J. Mans
    */
    class HcalHitReconstructor : public edm::EDProducer {
    public:
      explicit HcalHitReconstructor(const edm::ParameterSet& ps);
      virtual ~HcalHitReconstructor();
      virtual void beginRun(edm::Run&r, edm::EventSetup const & es);
      virtual void endRun(edm::Run&r, edm::EventSetup const & es);
      virtual void produce(edm::Event& e, const edm::EventSetup& c);
    private:      
      HcalSimpleRecAlgo reco_;
      HcalADCSaturationFlag* saturationFlagSetter_;
      HFTimingTrustFlag* HFTimingTrustFlagSetter_;
      HBHEStatusBitSetter* hbheFlagSetter_;
      HBHETimeProfileStatusBitSetter* hbheHSCPFlagSetter_;
      HBHETimingShapedFlagSetter* hbheTimingShapedFlagSetter_;
      HBHEPulseShapeFlagSetter *hbhePulseShapeFlagSetter_;
      HcalHFStatusBitFromDigis*   hfdigibit_;
      HcalHF_S9S1algorithm*       hfS9S1_;
      HcalHF_S9S1algorithm*       hfS8S1_;
      HcalHF_PETalgorithm*        hfPET_;
 
      DetId::Detector det_;
      int subdet_;
      HcalOtherSubdetector subdetOther_;
      edm::InputTag inputLabel_;
      //std::vector<std::string> channelStatusToDrop_;
      bool correctTiming_; // turn on/off Ken Rossato's algorithm to fix timing
      bool setNoiseFlags_; // turn on/off basic noise flags
      bool setHSCPFlags_;  // turn on/off HSCP noise flags
      bool setSaturationFlags_; // turn on/off flag indicating ADC saturation
      bool setTimingTrustFlags_; // turn on/off HF timing uncertainty flag 
      bool setPulseShapeFlags_; //  turn on/off HBHE fit-based noise flags
      bool dropZSmarkedPassed_; // turn on/off dropping of zero suppression marked and passed digis

      int firstAuxTS_;
  
      // legacy parameters for config-set values compatibility 
      // to be removed after 4_2_0...
      int firstSample_;
      int samplesToAdd_;
      bool tsFromDB_;
      
      HcalRecoParams* paramTS;  // firstSample & sampleToAdd from DB  
    };

#endif
