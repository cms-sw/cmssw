#ifndef CSCRecHitD_CSCPeakBinOfStripPulse_h
#define CSCRecHitD_CSCPeakBinOfStripPulse_h

/** \class CSCPeakBinOfStripPulse
 *
 * Class used to identify ADC peak pulse height and T_max
 * on strips in the endcap muon CSCs.  
 *
 * \author Dominique Fortin
 *
 */

#include <FWCore/ParameterSet/interface/ParameterSet.h>

class CSCStripDigi;

class CSCPeakBinOfStripPulse 
{
  
 public:

  /// configurable parameters
  CSCPeakBinOfStripPulse(const edm::ParameterSet & ps);
  ~CSCPeakBinOfStripPulse();

  /// This finds the strip seed for the cluster, that is the strip with maximum deposition
  bool peakAboveBaseline( const CSCStripDigi& digi, float& hmax, int& tmax, float* height ) const;

  /// Find the Strip pulseheight baseline 
  float baseline( const CSCStripDigi& digi ) const;

 private:


};

#endif

