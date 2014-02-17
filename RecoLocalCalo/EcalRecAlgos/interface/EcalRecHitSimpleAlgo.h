#ifndef RecoLocalCalo_EcalRecAlgos_EcalRecHitSimpleAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalRecHitSimpleAlgo_HH

/** \class EcalRecHitSimpleAlgo
  *  Simple algoritm to make rechits from uncalibrated rechits
  *
  *  $Id: EcalRecHitSimpleAlgo.h,v 1.17 2012/02/17 21:28:12 franzoni Exp $
  *  $Date: 2012/02/17 21:28:12 $
  *  $Revision: 1.17 $
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

    // Now fill flags

    bool good=true;
    
    if ( uncalibRH.checkFlag(EcalUncalibratedRecHit::kLeadingEdgeRecovered) ){
            rh.setFlag(EcalRecHit::kLeadingEdgeRecovered);
            good=false; 
    } 
    if ( uncalibRH.checkFlag(EcalUncalibratedRecHit::kSaturated) ) { 
            // leading edge recovery failed - still keep the information 
            // about the saturation and do not flag as dead 
            rh.setFlag(EcalRecHit::kSaturated); 
            good=false;
    } 
    if( uncalibRH.isSaturated() ) {
            rh.setFlag(EcalRecHit::kSaturated);
            good=false;
    } 
    if ( uncalibRH.checkFlag(EcalUncalibratedRecHit::kOutOfTime) ) {
            rh.setFlag(EcalRecHit::kOutOfTime) ;
	    good=false;   
    } 
    if ( uncalibRH.checkFlag(EcalUncalibratedRecHit::kPoorReco) ) {
            rh.setFlag(EcalRecHit::kPoorReco);
            good=false;
    }
    if ( uncalibRH.checkFlag( EcalUncalibratedRecHit::kHasSwitchToGain6 ) ) {
            rh.setFlag(EcalRecHit::kHasSwitchToGain6);
    }
    if (uncalibRH.checkFlag( EcalUncalibratedRecHit::kHasSwitchToGain1 ) ) {
            rh.setFlag(EcalRecHit::kHasSwitchToGain1);
    }
    
    if (good) rh.setFlag(EcalRecHit::kGood);
    return rh;
  }

private:
  float adcToGeVConstant_;
  bool  adcToGeVConstantIsSet_;

};
#endif
