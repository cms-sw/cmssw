#ifndef CSCRecRecHitD_CSCRecoConditions_h
#define CSCRecRecHitD_CSCRecoConditions_h

/**
 * \class CSCRecoConditions
 *
 * Wrap CSCConditions class for use in CSC local reconstruction, in analogy with wrapper classes
 * Rick uses in CSCDigitizer.
 *
 * CSCConditions encapsulates the conditions data (e.g. calibration data) from the database.
 *
 * \author Tim Cox
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/CSCObjects/interface/CSCConditions.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

class CSCRecoConditions 
{
 public:

  // Passed a PSet just in case we need to configure in some way
  explicit CSCRecoConditions(const edm::ParameterSet & pset);
  ~CSCRecoConditions();

  /// fetch the cond data from the database
  void initializeEvent(const edm::EventSetup & es);

  /// channels count from 1
  float gain(const CSCDetId & id, int channel) const { 
     return theConditions.gain(id, channel);}

  /// return average gain over entire CSC system
  float averageGain() const { 
     return theConditions.averageGain(); }

  /// return gain weight for given strip channel
  ///
  /// WARNING - expects ME11 detId for both ME1b (channels 1-64) AND for ME1a (channels 65-80)
  float stripWeight( const CSCDetId& id, int channel ) const;

  ///  calculate gain weights for all strips in a CSC layer 
  /// - filled into C-array which caller must have allocated.
  void stripWeights( const CSCDetId& id, float* weights ) const;

  /// in ADC counts
  float pedestal(const CSCDetId & id, int channel) const { 
     return theConditions.pedestal(id, channel);}

  float pedestalSigma(const CSCDetId & id, int channel) const { 
     return theConditions.pedestalSigma(id, channel);}

  /// fill expanded noise matrix for 3 neighbouring strip channels as linear vector (must be allocated by caller)
  void noiseMatrix( const CSCDetId& id, int channel, std::vector<float>& nme ) const;

  /// fill crosstalk information for 3 neighbouring strip channels as linear vector (must be allocated by caller)
  void crossTalk( const CSCDetId& id, int centralStrip, std::vector<float>& xtalks) const;

  /// Get bad strip word
  const std::bitset<80>& badStripWord( const CSCDetId& id ) const {
    return theConditions.badStripWord( id );
  }

  /// Get bad wiregroup word
  const std::bitset<112>& badWireWord( const CSCDetId& id ) const {
    return theConditions.badWireWord( id );
  }

 private:

  CSCConditions theConditions;
};

#endif
