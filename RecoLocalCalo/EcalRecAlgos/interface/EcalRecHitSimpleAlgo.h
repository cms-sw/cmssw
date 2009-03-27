#ifndef RecoLocalCalo_EcalRecAlgos_EcalRecHitSimpleAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalRecHitSimpleAlgo_HH

/** \class EcalRecHitSimpleAlgo
  *  Simple algoritm to make rechits from uncalibrated rechits
  *
  *  $Id: EcalRecHitSimpleAlgo.h,v 1.2 2008/10/17 10:16:55 meridian Exp $
  *  $Date: 2008/10/17 10:16:55 $
  *  $Revision: 1.2 $
  *  \author Shahram Rahatlou, University of Rome & INFN, March 2006
  */

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitAbsAlgo.h"
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
                                const float& timeIntercalib = 0) const {

    if(!adcToGeVConstantIsSet_) {
      std::cout << "EcalRecHitSimpleAlgo::makeRecHit: adcToGeVConstant_ not set before calling this method!" << 
                   " will use -1 and produce bogus rechits!" << std::endl;
    }

    float clockToNsConstant = 25;
    float energy = uncalibRH.amplitude()*adcToGeVConstant_*intercalibConstant;
    float time   = uncalibRH.jitter() * clockToNsConstant + timeIntercalib;
    if (uncalibRH.isSaturated())
      time = EcalRecHit::kSATURATED;

    return EcalRecHit( uncalibRH.id(), energy, time );
  }

private:
  float adcToGeVConstant_;
  bool  adcToGeVConstantIsSet_;

};
#endif
