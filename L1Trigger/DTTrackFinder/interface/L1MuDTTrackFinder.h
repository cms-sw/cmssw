//-------------------------------------------------
//
/**  \class L1MuDTTrackFinder
 *
 *   L1 barrel Muon Trigger Track Finder (MTTF)
 *
 *   The barrel MTTF consists of:
 *      - 72 Sector Processors (SP), 
 *      - 12 Eta Processors,
 *      - 12 Wedge Sorters (WS) and
 *      -  1 DT Muon Sorter (MS)
 *
 *
 *
 *   N. Neumeister            CERN EP
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef L1MUDT_TRACK_FINDER_H
#define L1MUDT_TRACK_FINDER_H

//---------------
// C++ Headers --
//---------------

#include <vector>
#include <memory>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
class L1MuDTChambPhContainer;
class L1MuDTTFConfig;
class L1MuDTSecProcMap;
class L1MuDTSecProcId;
class L1MuDTSectorProcessor;
class L1MuDTEtaProcessor;
class L1MuDTWedgeSorter;
class L1MuDTMuonSorter;
class L1MuDTTrackCand;
class L1MuRegionalCand;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTTrackFinder {
public:
  /// container for muon candidates
  typedef std::vector<L1MuRegionalCand>::const_iterator TFtracks_const_iter;
  typedef std::vector<L1MuRegionalCand>::iterator TFtracks_iter;

  /// constructor
  L1MuDTTrackFinder(const edm::ParameterSet& ps, edm::ConsumesCollector&& iC);

  /// destructor
  ~L1MuDTTrackFinder();

  /// build the structure of the barrel MTTF
  void setup(edm::ConsumesCollector&& iC);

  /// run the barrel MTTF
  void run(const edm::Event& e, const edm::EventSetup& c);

  /// reset the barrel MTTF
  void reset();

  /// get a pointer to a Sector Processor
  const L1MuDTSectorProcessor* sp(const L1MuDTSecProcId&) const;

  /// get a pointer to an Eta Processor, index [0-11]
  inline const L1MuDTEtaProcessor* ep(int id) const { return m_epvec[id].get(); }

  /// get a pointer to a Wedge Sorter, index [0-11]
  inline const L1MuDTWedgeSorter* ws(int id) const { return m_wsvec[id].get(); }

  /// get a pointer to the DT Muon Sorter
  inline const L1MuDTMuonSorter* ms() const { return m_ms.get(); }

  /// get number of muon candidates found by the barrel MTTF
  int numberOfTracks();

  TFtracks_const_iter begin();

  TFtracks_const_iter end();

  void clear();

  /// get number of muon candidates found by the barrel MTTF at a given bx
  int numberOfTracks(int bx);

  /// return configuration
  const L1MuDTTFConfig* config() const { return m_config.get(); }

  std::vector<L1MuDTTrackCand>& getcache0() { return _cache0; }

  std::vector<L1MuRegionalCand>& getcache() { return _cache; }

private:
  std::vector<L1MuDTTrackCand> _cache0;
  std::vector<L1MuRegionalCand> _cache;
  std::unique_ptr<L1MuDTSecProcMap> m_spmap;                 ///< Sector Processors
  std::vector<std::unique_ptr<L1MuDTEtaProcessor>> m_epvec;  ///< Eta Processors
  std::vector<std::unique_ptr<L1MuDTWedgeSorter>> m_wsvec;   ///< Wedge Sorters
  std::unique_ptr<L1MuDTMuonSorter> m_ms;                    ///< DT Muon Sorter
  edm::EDGetTokenT<L1MuDTChambPhContainer> m_DTDigiToken;

  std::unique_ptr<L1MuDTTFConfig> m_config;  ///< Track Finder configuration
};

#endif
