#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecAbsAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecAbsAlgo_HH

/** \class EcalUncalibRecHitRecAbsAlgo
  *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using a weights method
  *
  *  $Id: EcalUncalibRecHitRecAbsAlgo.h,v 1.1 2005/10/25 09:10:01 rahatlou Exp $
  *  $Date: 2005/10/25 09:10:01 $
  *  $Revision: 1.1 $
  *  \author R. Bruneliere - A. Zabi
  */

#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include <vector>
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"

template<class C> class EcalUncalibRecHitRecAbsAlgo
{
 public:
  enum { nWeightsRows = 3, iAmplitude = 0, iPedestal = 1, iTime = 2 };

  /// Constructor
  //EcalUncalibRecHitRecAbsAlgo() { };

  /// Destructor
  //virtual ~EcalUncalibRecHitRecAbsAlgo() { };

  /// make rechits from dataframes

  virtual EcalUncalibratedRecHit makeRecHit(const C& dataFrame, 
					    const std::vector<double>& pedestals,
					    const std::vector<double>& gainRatios,
					    const std::vector<HepMatrix>& weights, 
					    const std::vector<HepSymMatrix>& chi2Matrix) = 0;

};
#endif
