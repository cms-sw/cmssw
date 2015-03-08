#ifndef ZDCHITRECONSTRUCTOR_H 
#define ZDCHITRECONSTRUCTOR_H 1

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/ZdcSimpleRecAlgo.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHFStatusBitFromRecHits.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHFStatusBitFromDigis.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalLongRecoParams.h"
#include "CondFormats/HcalObjects/interface/HcalLongRecoParam.h" 
#include "RecoLocalCalo/HcalRecAlgos/interface/HBHEStatusBitSetter.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalTimingCorrector.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HBHETimeProfileStatusBitSetter.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HBHETimingShapedFlag.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalADCSaturationFlag.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HFTimingTrustFlag.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

class HcalTopology;

    /** \class ZdcHitReconstructor
	
    \author E. Garcia - CSU
    ** Based on HcalSimpleReconstructor.h by J. Mans
    */
    class ZdcHitReconstructor : public edm::stream::EDProducer<> {
    public:
      explicit ZdcHitReconstructor(const edm::ParameterSet& ps);
      virtual ~ZdcHitReconstructor();
      virtual void beginRun(edm::Run const&r, edm::EventSetup const & es) override final;
      virtual void endRun(edm::Run const&r, edm::EventSetup const & es) override final;
      virtual void produce(edm::Event& e, const edm::EventSetup& c) override final;
    private:      
      ZdcSimpleRecAlgo reco_;
      HcalADCSaturationFlag* saturationFlagSetter_;
      HFTimingTrustFlag* HFTimingTrustFlagSetter_;
      HBHEStatusBitSetter* hbheFlagSetter_;
      HBHETimeProfileStatusBitSetter* hbheHSCPFlagSetter_;
      HBHETimingShapedFlagSetter* hbheTimingShapedFlagSetter_;
      HcalHFStatusBitFromRecHits* hfrechitbit_;
      HcalHFStatusBitFromDigis*   hfdigibit_;
 
      DetId::Detector det_;
      int subdet_;
      HcalOtherSubdetector subdetOther_;
      edm::EDGetTokenT<ZDCDigiCollection> tok_input_;
      //std::vector<std::string> channelStatusToDrop_;
      bool correctTiming_; // turn on/off Ken Rossato's algorithm to fix timing
      bool setNoiseFlags_; // turn on/off basic noise flags
      bool setHSCPFlags_;  // turn on/off HSCP noise flags
      bool setSaturationFlags_; // turn on/off flag indicating ADC saturation
      bool setTimingTrustFlags_; // turn on/off HF timing uncertainty flag 

      bool dropZSmarkedPassed_; // turn on/off dropping of zero suppression marked and passed digis
      std::vector<int> AuxTSvec_;
    // new lowGainEnergy variables
      int lowGainOffset_;
      double lowGainFrac_;
      
      HcalLongRecoParams* myobject; //noiseTS and signalTS from db
      HcalTopology *theTopology;
    };

#endif
