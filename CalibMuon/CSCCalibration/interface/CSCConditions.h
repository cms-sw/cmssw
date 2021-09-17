#ifndef CSCCalibration_CSCConditions_h
#define CSCCalibration_CSCConditions_h

#include "CondFormats/DataRecord/interface/CSCChamberTimeCorrectionsRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBChipSpeedCorrectionRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBGasGainCorrectionRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCBadStripsRcd.h"
#include "CondFormats/DataRecord/interface/CSCBadWiresRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperRecord.h"
#include "CalibMuon/CSCCalibration/interface/CSCIndexerRecord.h"
#include "CondFormats/DataRecord/interface/CSCBadChambersRcd.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include <bitset>
#include <vector>

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
 * in CSCStripDigi (and CSCWireDigi) after internal corrections within
 * CSCRawToDigi.
 *
 * The input CSCDetId is also 'geometric channel level' i.e. ME11A has its own
 * CSCDetId even in the ganged case,
 *
 * Ganged ME1a channels are 1-16 (and unganged, of course, 1-48)
 *
 * From CMSSW 61X, this class also handles separate algorithm versions for
 * indexing the conditions data and for mapping between online and offline
 * channel labelling.
 */

class CSCConditions {
public:
  explicit CSCConditions(const edm::ParameterSet &ps, edm::ConsumesCollector);
  ~CSCConditions();

  /// fetch database content via EventSetup
  void initializeEvent(const edm::EventSetup &es);

  /// gain per channel
  float gain(const CSCDetId &detId, int channel) const;
  /// overall calibration precision
  float gainSigma(const CSCDetId &detId, int channel) const { return 0.005; }

  /// static ped in ADC counts
  float pedestal(const CSCDetId &detId, int channel) const;
  /// static ped rms in ADC counts
  float pedestalSigma(const CSCDetId &detId, int channel) const;

  /// crosstalk slope for left and right
  float crosstalkSlope(const CSCDetId &detId, int channel, bool leftRight) const;
  /// crosstalk intercept for left and right
  float crosstalkIntercept(const CSCDetId &detId, int channel, bool leftRight) const;

  /// raw noise matrix (unscaled short int elements)
  const CSCDBNoiseMatrix::Item &noiseMatrix(const CSCDetId &detId, int channel) const;

  /// fill vector (dim 12, must be allocated by caller) with noise matrix
  /// elements (scaled to float)
  void noiseMatrixElements(const CSCDetId &id, int channel, std::vector<float> &me) const;

  /// fill vector (dim 4, must be allocated by caller) with crosstalk sl, il,
  /// sr, ir
  void crossTalk(const CSCDetId &id, int channel, std::vector<float> &ct) const;

  /// chip speed correction in ns given detId (w/layer) and strip channel
  float chipCorrection(const CSCDetId &detId, int channel) const;

  /// chamber timing correction in ns given detId of chamber
  float chamberTimingCorrection(const CSCDetId &detId) const;

  /// anode bx offset in bx given detId of chamber
  float anodeBXoffset(const CSCDetId &detId) const;

  /// bad strip channel word for a CSCLayer - 1 bit per channel
  const std::bitset<112> &badStripWord() const { return badStripWord_; }

  /// bad wiregroup channel word for a CSCLayer - 1 bit per channel
  const std::bitset<112> &badWireWord() const { return badWireWord_; }

  /// the offline CSCDetId of current bad channel words
  const CSCDetId &idOfBadChannelWords() const { return idOfBadChannelWords_; }

  void print() const;

  /// Is the gven chamber flagged as bad?
  bool isInBadChamber(const CSCDetId &id) const;

  /// did we request reading bad channel info from db?
  bool readBadChannels() const { return readBadChannels_; }

  /// did we request reading bad chamber info from db?
  bool readBadChambers() const { return readBadChambers_; }

  /// did we request reading timing correction info from db?
  bool useTimingCorrections() const { return useTimingCorrections_; }

  /// Fill bad channel words - one for strips, one for wires, for an offline
  /// CSCDetId
  void fillBadChannelWords(const CSCDetId &id);

  /// average gain over entire CSC system (logically const although must be
  /// cached here).
  float averageGain() const;

  /// gas gain correction as a function of detId (w/layer), strip, and wire
  /// channels
  float gasGainCorrection(const CSCDetId &detId, int strip, int wire) const;

  /// did we request reading gas gain correction info from db?
  bool useGasGainCorrections() const { return useGasGainCorrections_; }

  /// feedthrough for external access
  int channelFromStrip(const CSCDetId &id, int geomStrip) const;
  int rawStripChannel(const CSCDetId &id, int geomChannel) const;

private:
  /// fill bad channel words for offline id
  void fillBadStripWord(const CSCDetId &id);
  void fillBadWireWord(const CSCDetId &id);
  /// Set id for current content of bad channel words - this is offline id i.e.
  /// separate for ME11A & ME11B
  void setIdOfBadChannelWords(const CSCDetId &id) { idOfBadChannelWords_ = id; }

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

  //EventSetup Tokens for Handles
  edm::ESGetToken<CSCDBGains, CSCDBGainsRcd> gainsToken_;
  edm::ESGetToken<CSCDBCrosstalk, CSCDBCrosstalkRcd> crosstalkToken_;
  edm::ESGetToken<CSCDBPedestals, CSCDBPedestalsRcd> pedestalsToken_;
  edm::ESGetToken<CSCDBNoiseMatrix, CSCDBNoiseMatrixRcd> noiseMatrixToken_;
  edm::ESGetToken<CSCBadStrips, CSCBadStripsRcd> badStripsToken_;
  edm::ESGetToken<CSCBadWires, CSCBadWiresRcd> badWiresToken_;
  edm::ESGetToken<CSCBadChambers, CSCBadChambersRcd> badChambersToken_;
  edm::ESGetToken<CSCDBChipSpeedCorrection, CSCDBChipSpeedCorrectionRcd> chipCorrectionsToken_;
  edm::ESGetToken<CSCChamberTimeCorrections, CSCChamberTimeCorrectionsRcd> chamberTimingCorrectionsToken_;
  edm::ESGetToken<CSCDBGasGainCorrection, CSCDBGasGainCorrectionRcd> gasGainCorrectionsToken_;
  edm::ESGetToken<CSCIndexerBase, CSCIndexerRecord> indexerToken_;
  edm::ESGetToken<CSCChannelMapperBase, CSCChannelMapperRecord> mapperToken_;

  // logical flags controlling some conditions data usage

  bool readBadChannels_;        // flag whether or not to even attempt reading bad
                                // channel info from db
  bool readBadChambers_;        // flag whether or not to even attempt reading bad
                                // chamber info from db
  bool useTimingCorrections_;   // flag whether or not to even attempt reading
                                // timing correction info from db
  bool useGasGainCorrections_;  // flag whether or not to even attempt reading
                                // gas-gain correction info from db

  // Cache bad channel content for current CSC layer
  CSCDetId idOfBadChannelWords_;
  std::bitset<112> badStripWord_;
  std::bitset<112> badWireWord_;

  mutable float theAverageGain;  // average over entire system, subject to some
                                 // constraints!

  edm::ESWatcher<CSCDBGainsRcd> gainsWatcher_;
  //@@ remove until we have real information to use
  //  edm::ESWatcher<CSCBadStripsRcd> badStripsWatcher_;
  //  edm::ESWatcher<CSCBadWiresRcd> badWiresWatcher_;

  // Total number of CSC layers in the system, with full ME42 installed.
  enum elayers { MAX_LAYERS = 3240 };
};

#endif
