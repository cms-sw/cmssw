//-------------------------------------------------
//
/**  \class L1MuDTEtaProcessor
 *
 *   Eta Processor:
 *
 *   An Eta Processor consists of :
 *    - one receiver unit,
 *    - one Eta Track Finder (ETF) and 
 *    - one Eta Matching Unit (EMU)
 *
 *
 *
 *   N. Neumeister            CERN EP
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef L1MUDT_ETA_PROCESSOR_H
#define L1MUDT_ETA_PROCESSOR_H

//---------------
// C++ Headers --
//---------------

#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTAddressArray.h"
class L1MuDTChambThContainer;
class L1MuDTTrackSegEta;
class L1MuDTTrackFinder;
class L1MuDTTrack;
class L1MuDTEtaPatternLut;
class L1MuDTQualPatternLut;
class L1MuDTTFMasks;
class L1MuDTTFMasksRcd;
class L1MuDTEtaPatternLutRcd;
class L1MuDTQualPatternLutRcd;
//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTEtaProcessor {
public:
  /// constructor
  L1MuDTEtaProcessor(const L1MuDTTrackFinder&, int id, edm::ConsumesCollector iC);

  /// destructor
  virtual ~L1MuDTEtaProcessor();

  /// return Eta Processor identifier (0-11)
  inline int id() const { return m_epid; }

  /// run the Eta Processor
  virtual void run(int bx, const edm::Event& e, const edm::EventSetup& c);

  /// reset the Eta Processor
  virtual void reset();

  /// print muon candidates found by the Eta Processor
  void print() const;

  /// return reference to barrel MTTF
  inline const L1MuDTTrackFinder& tf() const { return m_tf; }

  /// return eta values, index [0,11]
  inline int eta(int id) const { return m_eta[id]; }

  /// return fine bit, index [0,11]
  inline bool fineBit(int id) const { return m_fine[id]; }

private:
  /// receive data (eta trigger primitives)
  void receiveData(int bx, const edm::Event& e, const edm::EventSetup& c);

  /// receive addresses (from 6 Sector Processors)
  void receiveAddresses();

  /// run Eta Track Finder (ETF)
  void runEtaTrackFinder(const edm::EventSetup& c);

  /// run Eta Matching Unit (EMU)
  void runEtaMatchingUnit(const edm::EventSetup& c);

  /// assign eta and etaFineBit
  void assign();

  /// get quality code; id [0,26], stat [1,3]
  static int quality(int id, int stat);

private:
  const L1MuDTTrackFinder& m_tf;
  int m_epid;

  int m_mask;

  int m_eta[12];
  bool m_fine[12];

  std::vector<int> m_foundPattern;
  int m_pattern[12];

  int m_address[12];
  L1MuDTTrack* m_TrackCand[12];
  L1MuDTTrack* m_TracKCand[12];
  std::vector<const L1MuDTTrackSegEta*> m_tseta;
  edm::EDGetTokenT<L1MuDTChambThContainer> m_DTDigiToken;

  edm::ESGetToken<L1MuDTEtaPatternLut, L1MuDTEtaPatternLutRcd> theEtaToken;
  edm::ESGetToken<L1MuDTQualPatternLut, L1MuDTQualPatternLutRcd> theQualToken;
  edm::ESGetToken<L1MuDTTFMasks, L1MuDTTFMasksRcd> theMsksToken;
  edm::ESHandle<L1MuDTEtaPatternLut> theEtaPatternLUT;    // ETF look-up table
  edm::ESHandle<L1MuDTQualPatternLut> theQualPatternLUT;  // EMU look-up tables
  edm::ESHandle<L1MuDTTFMasks> msks;
};

#endif
