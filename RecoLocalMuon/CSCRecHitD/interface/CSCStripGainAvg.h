#ifndef CSCRecHitD_CSCStripGainAvg_h
#define CSCRecHitD_CSCStripGainAvg_h

/** \class CSCStripGainAvg
 *
 * This routine finds the average global gain for the whole CSC system, which is
 * needed to compute the correction weight for the strip gains.
 *
 * \author Dominique Fortin - UCR
 */

#include <CondFormats/CSCObjects/interface/CSCDBGains.h>
#include <CondFormats/DataRecord/interface/CSCDBGainsRcd.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>


class CSCStripGainAvg {

 public:

  /// configurable parameters
  explicit CSCStripGainAvg(const edm::ParameterSet & ps);  
  ~CSCStripGainAvg();

  // Member functions

  /// Load in the gains, X-talk and noise matrix and store in memory
  void setCalibration( const CSCDBGains* gains ) { Gains = gains; }
 
  /// Computes the average gain for the whole CSC system.
  float getStripGainAvg();

 private:

  // Store in memory Gains
  const CSCDBGains         * Gains;

};

#endif

