#ifndef CSCObjects_CSCConditions_h
#define CSCObjects_CSCConditions_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
class CSCDBGains;
class CSCDBPedestals;
class CSCDBCrosstalk;

/*  Encapsulates a user interface into the CSC conditions
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

  const CSCDBNoiseMatrix::Item &
  noiseMatrix(const CSCDetId&detId, int channel) const;

  void print() const;

private:

  const CSCDBNoiseMatrix * theNoiseMatrix;
  const CSCDBGains * theGains;
  const CSCDBPedestals * thePedestals;
  const CSCDBCrosstalk * theCrosstalk;

};

#endif


