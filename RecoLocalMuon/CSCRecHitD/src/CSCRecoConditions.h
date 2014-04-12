#ifndef CSCRecRecHitD_CSCRecoConditions_h
#define CSCRecRecHitD_CSCRecoConditions_h

/**
 * \class CSCRecoConditions
 *
 * Wrap CSCConditions class for use in CSC local reconstruction, in analogy with wrapper classes
 * Rick uses in CSCDigitizer.
 *
 * CSCConditions encapsulates the conditions data (e.g. calibration data) from the database
 * and presents it as CSCRecHitD requires (and that is somewhat historical!)
 *
 * All functions in public interface accept CSCDetId for ME1A (i.e. ring 4) with channel
 * number 1-16 (and not the raw ME11 channel 65-80).
 *
 * \author Tim Cox - UC Davis
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibMuon/CSCCalibration/interface/CSCConditions.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

class CSCRecoConditions
{
 public:

  // Passed a PSet just in case we need to configure in some way
  explicit CSCRecoConditions(const edm::ParameterSet & pset);
  ~CSCRecoConditions();

  /// fetch the cond data from the database
  void initializeEvent(const edm::EventSetup & es);

  /// channels and geomstrips  count from 1

  /// return gain for given strip
  float gain( const CSCDetId& id, int geomStrip ) const;

  /// return average gain over entire CSC system
  float averageGain() const {
     return theConditions.averageGain(); }

  ///  calculate gain weights for all strips in a CSC layer (total in layer = nstrips)
  ///  this is averageGain()/gain for each strip filled into a C-array which caller must have allocated.
  ///  values are constrained to be in [0.5, 1.5]
  void stripWeights( const CSCDetId& id, short int nstrips, float* weights ) const;

  /// static pedestal in ADC counts for strip channel (e.g. 1-16 for ganged ME1a, 1-48 for unganged ME1a)
  float pedestal(const CSCDetId& id, int channel) const;

  /// sigma of static pedestal in ADC counts for strip channel (e.g. 1-16 for ganged ME1a, 1-48 for unganged ME1a)
  float pedestalSigma(const CSCDetId& id, int channel) const;

  /// fill expanded noise matrix for 3 neighbouring strips as linear vector (must be allocated by caller)
  /// Note that centralStrip is a 'geomStrip' and ranges 1-48 in ME1a.
  void noiseMatrix( const CSCDetId& id, int centralStrip, std::vector<float>& nme ) const;

  /// fill crosstalk information for 3 neighbouring strips as linear vector (must be allocated by caller)
  /// Note that centralStrip is a 'geomStrip' and e.g. always ranges 1-48 in ME1a.
  void crossTalk( const CSCDetId& id, int centralStrip, std::vector<float>& xtalks) const;

   // returns chip speed correction in ns given strio channel
  float chipCorrection( const CSCDetId & detId, int channel ) const;

   // returns chamber level timing correction (cable length and extra chamber correction) in ns
  float chamberTimingCorrection( const CSCDetId& id )const;

   // returns anode bx off for each chamber, used to correct anode times to 0 for collision muons
  float anodeBXoffset( const CSCDetId & detId )const;

  /// returns gas-gain correction
  float gasGainCorrection( const CSCDetId& id, int strip, int wireGroup ) const;

  /// Is a neighbour bad?
  bool nearBadStrip( const CSCDetId& id, int geomStrip ) const;

  /// Is the strip bad?
  bool badStrip( const CSCDetId& id, int geomStrip ) const;

  /// Get bad wiregroup word
  const std::bitset<112>& badWireWord( const CSCDetId& id ) const;

 private:

  /// return gain weight for given strip channel

  float stripWeight( const CSCDetId& id, int geomStrip ) const;

  CSCConditions theConditions;
};

#endif
