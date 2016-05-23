//-------------------------------------------------
//
/**  \class L1MuBMTrackFinder
 *
 *   L1 barrel Muon Trigger Track Finder (MTTF)
 *
 *   The barrel MTTF consists of:
 *      - 72 Sector Processors (SP),
 *      - 12 Eta Processors,
 *      - 12 Wedge Sorters (WS) and
 *      -  1 BM Muon Sorter (MS)
 *
 *
 *
 *   N. Neumeister            CERN EP
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef L1MUBM_TRACK_FINDER_H
#define L1MUBM_TRACK_FINDER_H

//---------------
// C++ Headers --
//---------------

#include <vector>
#include <iostream>
//----------------------
// Base Class Headers --
//----------------------

#include <map>

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMTrack.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

class L1MuBMTFConfig;
class L1MuBMSecProcMap;
class L1MuBMSecProcId;
class L1MuBMSectorProcessor;
class L1MuBMEtaProcessor;
class L1MuBMWedgeSorter;
class L1MuBMMuonSorter;
class BMTrackCand;
class L1MuRegionalCand;

//              ---------------------
//              -- Class Interface --
//              ---------------------


class L1MuBMTrackFinder {

  public:

    /// container for muon candidates
    typedef l1t::RegionalMuonCandBxCollection::const_iterator TFtracks_const_iter;
    typedef l1t::RegionalMuonCandBxCollection::iterator       TFtracks_iter;

    /// constructor
    L1MuBMTrackFinder(const edm::ParameterSet & ps, edm::ConsumesCollector && iC);

    /// destructor
    virtual ~L1MuBMTrackFinder();

    /// build the structure of the barrel MTTF
    void setup(edm::ConsumesCollector&& );

    /// run the barrel MTTF
    void run(const edm::Event& e, const edm::EventSetup& c);

    /// reset the barrel MTTF
    void reset();

    inline int setAdd(int ust, int rel_add);

    /// get a pointer to a Sector Processor
    const L1MuBMSectorProcessor* sp(const L1MuBMSecProcId&) const;

    /// get a pointer to an Eta Processor, index [0-11]
    inline const L1MuBMEtaProcessor* ep(int id) const { return m_epvec[id]; }

    /// get a pointer to a Wedge Sorter, index [0-11]
    inline const L1MuBMWedgeSorter* ws(int id) const { return m_wsvec[id]; }

    /// get a pointer to the BM Muon Sorter
    inline const L1MuBMMuonSorter* ms() const { return m_ms; }

    /// get number of muon candidates found by the barrel MTTF
    int numberOfTracks();

    TFtracks_const_iter begin(int bx);

    TFtracks_const_iter end(int bx);

    void clear();

    /// get number of muon candidates found by the barrel MTTF at a given bx
    int numberOfTracks(int bx);

    /// return configuration
    static const L1MuBMTFConfig* config() { return m_config; }

//    std::vector<BMTrackCand>& getcache0() { return _cache0; }

    l1t::RegionalMuonCandBxCollection& getcache() { return _cache; }
    l1t::RegionalMuonCandBxCollection& getcache0() { return _cache0; }

  private:

    /// run Track Finder and store candidates in cache
    virtual void reconstruct(const edm::Event& e, const edm::EventSetup& c) { reset(); run(e,c); }

  private:

    l1t::RegionalMuonCandBxCollection _cache0;
    l1t::RegionalMuonCandBxCollection _cache;
    L1MuBMSecProcMap*                m_spmap;        ///< Sector Processors
    std::vector<L1MuBMEtaProcessor*> m_epvec;        ///< Eta Processors
    std::vector<L1MuBMWedgeSorter*>  m_wsvec;        ///< Wedge Sorters
    L1MuBMMuonSorter*                m_ms;           ///< BM Muon Sorter

    static L1MuBMTFConfig*           m_config;       ///< Track Finder configuration

    std::map<int,int> eta_map ={
 {-32,  -118}, {-31,  -114},  {-30,  -111},  {-29,  -107},  {-28,  -103},  {-27,   -99},  {-26,   -96},  {-25,   -92}, {-24,   -88},
 {-23,   -84}, {-22,   -81}, {-21,   -77}, {-20,   -73}, {-19,   -69}, {-18,   -66}, {-17,   -62}, {-16,   -58}, {-15,   -54}, {-14,   -51},
 {-13,   -47}, {-12,   -43}, {-11,   -39}, {-10,   -36},  {-9,   -32},  {-8,   -28},  {-7,   -24},  {-6,   -21},  {-5,   -17},  {-4,   -13},
  {-3,    -9},  {-2,    -6},  {-1,    -2},  {0,     0},  {1,     6},  {2,     9},  {3,    13},  {4,    17},  {5,    21},  {6,    24},
  {7,    28},  {8,    32},  {9,    36},  {10,    39},  {11,    43},  {12,    47},  {13,    51},  {14,    54},  {15,    58},  {16,    62},
  {17,    66},  {18,    69},  {19,    73},  {20,    77},  {21,    81},  {22,    84},  {23,    88},  {24,    92},  {25,    96},
  {26,    99},  {27,   103},  {28,   107},  {29,   111},  {30,   114},  {31,   118}};


    edm::EDGetTokenT<L1MuDTChambPhContainer> m_DTDigiToken;


};

#endif
