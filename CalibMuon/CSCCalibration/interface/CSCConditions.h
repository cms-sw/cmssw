#ifndef CSCCalibration_CSCConditions_h
#define CSCCalibration_CSCConditions_h

#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "CondFormats/DataRecord/interface/CSCBadStripsRcd.h"
#include "CondFormats/DataRecord/interface/CSCBadWiresRcd.h"

#include <vector>
#include <bitset>

class CSCDBGains;
class CSCDBPedestals;
class CSCDBCrosstalk;
class CSCBadStrips;
class CSCBadWires;
class CSCBadChambers;
class CSCDBChipSpeedCorrection;
class CSCChamberTimeCorrections;
class CSCDBGasGainCorrection;
class CSCIndexerBase;
class CSCChannelMapperBase;

/**  Encapsulates a user interface into the CSC conditions
 *
 * \author Rick Wilkinson
 * \author Tim Cox
 *
 * Interfaces generally use "channels" which count from 1 and are 'geometric'
 * i.e. in the order matching local coordinates. This is the channel labelling
 * in CSCStripDigi (and CSCWireDigi) after internal corrections within CSCRawToDigi.
 *
 * The input CSCDetId is also 'geometric channel level' i.e. ME11A has its own CSCDetId
 * even in the ganged case,
 *
 * Ganged ME1a channels are 1-16 (and unganged, of course, 1-48)
 *
 * From CMSSW 61X, this class also handles separate algorithm versions for indexing
 * the conditions data and for mapping between online and offline channel labelling.
 */

class CSCConditions
{
public:
  explicit CSCConditions( const edm::ParameterSet& ps );
  ~CSCConditions();

  /// fetch database content via EventSetup
  void initializeEvent(const edm::EventSetup & es);

  /// gain per channel
  float gain(const CSCDetId & detId, int channel) const;
  /// overall calibration precision
  float gainSigma(const CSCDetId & detId, int channel) const {return 0.005;}

  /// static ped in ADC counts
  float pedestal(const CSCDetId & detId, int channel) const;
  /// static ped rms in ADC counts
  float pedestalSigma(const CSCDetId & detId, int channel) const;

  /// crosstalk slope for left and right
  float crosstalkSlope(const CSCDetId & detId, int channel, bool leftRight) const;
  /// crosstalk intercept for left and right
  float crosstalkIntercept(const CSCDetId & detId, int channel, bool leftRight) const;

  /// raw noise matrix (unscaled short int elements)
  const CSCDBNoiseMatrix::Item & noiseMatrix(const CSCDetId & detId, int channel) const;

  /// fill vector (dim 12, must be allocated by caller) with noise matrix elements (scaled to float)
  void noiseMatrixElements( const CSCDetId& id, int channel, std::vector<float>& me ) const;

  /// fill vector (dim 4, must be allocated by caller) with crosstalk sl, il, sr, ir
  void crossTalk( const CSCDetId& id, int channel, std::vector<float>& ct ) const;

  /// chip speed correction in ns given detId (w/layer) and strip channel
  float chipCorrection( const CSCDetId & detId, int channel ) const;

  /// chamber timing correction in ns given detId of chamber
  float chamberTimingCorrection( const CSCDetId & detId) const;

  /// anode bx offset in bx given detId of chamber
  float anodeBXoffset( const CSCDetId & detId) const;

  /// bad channel words per CSCLayer - 1 bit per channel
  const std::bitset<80>& badStripWord( const CSCDetId& id ) const;
  const std::bitset<112>& badWireWord( const CSCDetId& id ) const;

  void print() const;

  /// Is the gven chamber flagged as bad?
  bool isInBadChamber( const CSCDetId& id ) const;

  /// did we request reading bad channel info from db?
  bool readBadChannels() const { return readBadChannels_; }

  /// did we request reading bad chamber info from db?
  bool readBadChambers() const { return readBadChambers_; }

  /// did we request reading timing correction info from db?
  bool useTimingCorrections() const { return useTimingCorrections_; }

  /// fill bad channel words
  void fillBadStripWords();
  void fillBadWireWords();

  /// average gain over entire CSC system (logically const although must be cached here).
  float averageGain() const;

  /// gas gain correction as a function of detId (w/layer), strip, and wire channels
  float gasGainCorrection( const CSCDetId & detId, int strip, int wire ) const;

  /// did we request reading gas gain correction info from db?
  bool useGasGainCorrections() const { return useGasGainCorrections_; }

  /// feedthrough for external access
  int channelFromStrip(const CSCDetId& id, int geomStrip) const;
  int rawStripChannel( const CSCDetId& id, int geomChannel) const;

private:

// handles to conditions data

  edm::ESHandle<CSCDBGains> theGains;
  edm::ESHandle<CSCDBCrosstalk> theCrosstalk;
  edm::ESHandle<CSCDBPedestals> thePedestals;
  edm::ESHandle<CSCDBNoiseMatrix> theNoiseMatrix;
  edm::ESHandle<CSCBadStrips> theBadStrips;
  edm::ESHandle<CSCBadWires> theBadWires;
  edm::ESHandle<CSCBadChambers> theBadChambers;
  edm::ESHandle<CSCDBChipSpeedCorrection> theChipCorrections;
  edm::ESHandle<CSCChamberTimeCorrections> theChamberTimingCorrections;
  edm::ESHandle<CSCDBGasGainCorrection> theGasGainCorrections;

// handles to algorithm versions

  edm::ESHandle<CSCIndexerBase> indexer_;
  edm::ESHandle<CSCChannelMapperBase> mapper_;

// logical flags controlling some conditions data usage

  bool readBadChannels_; // flag whether or not to even attempt reading bad channel info from db
  bool readBadChambers_; // flag whether or not to even attempt reading bad chamber info from db
  bool useTimingCorrections_; // flag whether or not to even attempt reading timing correction info from db
  bool useGasGainCorrections_; // flag whether or not to even attempt reading gas-gain correction info from db

  // cache bad channel words once created
  std::vector< std::bitset<80> > badStripWords;
  std::vector< std::bitset<112> > badWireWords;

  mutable float theAverageGain; // average over entire system, subject to some constraints!

  edm::ESWatcher<CSCDBGainsRcd> gainsWatcher_;
  //@@ remove until we have real information to use
  //  edm::ESWatcher<CSCBadStripsRcd> badStripsWatcher_;
  //  edm::ESWatcher<CSCBadWiresRcd> badWiresWatcher_;

  // Total number of CSC layers in the system, with full ME42 installed.
  enum elayers{ MAX_LAYERS = 3240 };
};

#endif


