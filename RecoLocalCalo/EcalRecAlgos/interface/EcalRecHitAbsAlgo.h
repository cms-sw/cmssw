#ifndef RecoLocalCalo_EcalRecAlgos_EcalRecHitAbsAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalRecHitAbsAlgo_HH

/** \class EcalRecHitAbsAlgo
  *  Template algorithm to make rechits from uncalibrated rechits
  *
  *  $Id: EcalRecHitAbsAlgo.h,v 1.2 2009/03/27 17:04:12 ferriff Exp $
  *  $Date: 2009/03/27 17:04:12 $
  *  $Revision: 1.2 $
  *  \author Shahram Rahatlou, University of Rome & INFN, March 2006
  */

#include <vector>
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"

class EcalRecHitAbsAlgo
{
 public:

  /// Constructor
  //EcalRecHitAbsAlgo() { };

  /// Destructor
  //virtual ~EcalRecHitAbsAlgo() { };

  /// make rechits from dataframes

  virtual void setADCToGeVConstant(const float& value) = 0;
  virtual EcalRecHit makeRecHit(const EcalUncalibratedRecHit& uncalibRH, const float& intercalib, const float& timecalib, const uint32_t &flags) const = 0;

};
#endif
