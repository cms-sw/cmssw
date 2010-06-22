#ifndef RecoLocalCalo_EcalRecAlgos_EcalRecHitSimpleAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalRecHitSimpleAlgo_HH

/** \class EcalRecHitSimpleAlgo
  *  Simple algoritm to make rechits from uncalibrated rechits
  *
  *  $Id: EcalRecHitSimpleAlgo.h,v 1.10 2010/04/13 14:39:14 rahatlou Exp $
  *  $Date: 2010/04/13 14:39:14 $
  *  $Revision: 1.10 $
  *  \author Shahram Rahatlou, University of Rome & INFN, March 2006
  */

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitAbsAlgo.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "TMath.h"
#include <iostream>

class EcalRecHitSimpleAlgo : public EcalRecHitAbsAlgo {
 public:
  // default ctor
  EcalRecHitSimpleAlgo() {
    adcToGeVConstant_ = -1;
    adcToGeVConstantIsSet_ = false;
  }

  virtual void setADCToGeVConstant(const float& value) {
    adcToGeVConstant_ = value;
    adcToGeVConstantIsSet_ = true;
  }


  // destructor
  virtual ~EcalRecHitSimpleAlgo() { };

  /// Compute parameters
  virtual EcalRecHit makeRecHit(const EcalUncalibratedRecHit& uncalibRH,
                                const float& intercalibConstant,
                                const float& timeIntercalib = 0,
                                const uint32_t& flags = 0) const {

    if(!adcToGeVConstantIsSet_) {
      std::cout << "EcalRecHitSimpleAlgo::makeRecHit: adcToGeVConstant_ not set before calling this method!" << 
                   " will use -1 and produce bogus rechits!" << std::endl;
    }

    float clockToNsConstant = 25;
    float energy = uncalibRH.amplitude()*adcToGeVConstant_*intercalibConstant;
    float time   = uncalibRH.jitter() * clockToNsConstant + timeIntercalib;

    EcalRecHit rh( uncalibRH.id(), energy, time );
    rh.setChi2( uncalibRH.chi2() );
    rh.setOutOfTimeEnergy( uncalibRH.outOfTimeEnergy() * adcToGeVConstant_ * intercalibConstant );
    rh.setOutOfTimeChi2( uncalibRH.outOfTimeChi2() );
    rh.setTimeError(uncalibRH.jitterErrorBits());

    // Now fill both recoFlag and the new flagBits
    uint32_t flagbits(0);
    uint32_t recoFlag = flags; // so far contains only v_DB_reco_flags_[ statusCode ] from the worker
                               // for the time being ignore them in flagBits

    if ( uncalibRH.recoFlag() == EcalUncalibratedRecHit::kLeadingEdgeRecovered ) { 
            recoFlag = EcalRecHit::kLeadingEdgeRecovered; 
            flagbits |=  (0x1 << EcalRecHit::kLeadingEdgeRecovered); 

    } else if ( uncalibRH.recoFlag() == EcalUncalibratedRecHit::kSaturated ) { 
            // leading edge recovery failed - still keep the information 
            // about the saturation and do not flag as dead 
            recoFlag = EcalRecHit::kSaturated; 
            // and at some point may try the recovery with the neighbours 
            flagbits |=  (0x1 << EcalRecHit::kSaturated); 

    } else if( uncalibRH.isSaturated() ) {
            recoFlag = EcalRecHit::kSaturated;
            flagbits |=  (0x1 << EcalRecHit::kSaturated); 

    } else if ( uncalibRH.recoFlag() == EcalUncalibratedRecHit::kOutOfTime ) {
            recoFlag =  EcalRecHit::kOutOfTime;
            flagbits |=  (0x1 << EcalRecHit::kOutOfTime); 

    } else if ( uncalibRH.recoFlag() == EcalUncalibratedRecHit::kFake ) {
            recoFlag =  EcalRecHit::kFake;
            flagbits |=  (0x1 << EcalRecHit::kFake); 
    }

    // now set both the reco flag and the new flagBits_
    rh.setRecoFlag( recoFlag );
    rh.setFlagBits( flagbits );

    return rh;
  }

private:
  float adcToGeVConstant_;
  bool  adcToGeVConstantIsSet_;

};
#endif
