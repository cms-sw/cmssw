#ifndef HCALHITRECONSTRUCTOR_H 
#define HCALHITRECONSTRUCTOR_H 1

#include "FWCore/Framework/interface/EDProducer.h"
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
#include "CondFormats/HcalObjects/interface/HcalFlagHFDigiTimeParams.h"

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
	
    $Date: 2013/05/08 23:20:44 $
    $Revision: 1.23 $
    \author J. Temple & E. Yazgan
    ** Based on HcalSimpleReconstructor.h by J. Mans
    */

class HcalTopology;

    class HcalHitReconstructor : public edm::EDProducer {
    public:
      explicit HcalHitReconstructor(const edm::ParameterSet& ps);
      virtual ~HcalHitReconstructor();
      virtual void beginRun(edm::Run const&r, edm::EventSetup const & es) override final;
      virtual void endRun(edm::Run const&r, edm::EventSetup const & es) override final;
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
      int firstSample_;
      int samplesToAdd_;
      bool tsFromDB_;
      bool recoParamsFromDB_;
      bool digiTimeFromDB_;


      // switch on/off leakage (to pre-sample) correction
      bool useLeakCorrection_;
      
      HcalRecoParams* paramTS;  // firstSample & sampleToAdd from DB  
      const HcalFlagHFDigiTimeParams* HFDigiTimeParams; // HF DigiTime parameters

      HcalTopology *theTopology;
    };

#endif
