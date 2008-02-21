#ifndef CSCObjects_CSCConditions_h
#define CSCObjects_CSCConditions_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include <vector>

class CSCDBGains;
class CSCDBPedestals;
class CSCDBCrosstalk;

/**  Encapsulates a user interface into the CSC conditions
 *
 * \author Rick Wilkinson
 * \author Tim Cox
 */

class CSCConditions
{
public:
  CSCConditions();
  ~CSCConditions();

  /// fetch the maps from the database
  void initializeEvent(const edm::EventSetup & es);

  /// channels count from 1
  float gain(const CSCDetId & detId, int channel) const;
  /// total calibration precision
  float gainSigma(const CSCDetId & detId, int channel) const {return 0.005;}

  /// in ADC counts
  float pedestal(const CSCDetId & detId, int channel) const;
  float pedestalSigma(const CSCDetId & detId, int channel) const;

  float crosstalkSlope(const CSCDetId&detId, int channel, bool leftRight) const;
  float crosstalkIntercept(const CSCDetId&detId, int channel, bool leftRight) const;

  /// return raw noise matrix (unscaled short int elements)
  const CSCDBNoiseMatrix::Item & noiseMatrix(const CSCDetId&detId, int channel) const;

  /// fill vector (dim 12, must be allocated by caller) with noise matrix elements (scaled to float)
  void noiseMatrixElements( const CSCDetId& id, int channel, std::vector<float>& me ) const;

  /// fill vector (dim 4, must be allocated by caller) with crosstalk sl, il, sr, ir
  void crossTalk( const CSCDetId& id, int channel, std::vector<float>& ct ) const;

  void print() const;

  /// average gain over entire CSC system (logically const although must be cached here).
  float averageGain() const;

private:

  const CSCDBNoiseMatrix * theNoiseMatrix;
  const CSCDBGains * theGains;
  const CSCDBPedestals * thePedestals;
  const CSCDBCrosstalk * theCrosstalk;

  mutable float theAverageGain; // average over entire system, subject to some constraints!
};

#endif


