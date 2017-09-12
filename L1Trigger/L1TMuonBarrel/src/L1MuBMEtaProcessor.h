//-------------------------------------------------
//
/**  \class L1MuBMEtaProcessor
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
#ifndef L1MUBM_ETA_PROCESSOR_H
#define L1MUBM_ETA_PROCESSOR_H

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

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMAddressArray.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsRcd.h"

#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMTQualPatternLut.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMTEtaPatternLut.h"

class L1MuBMTrackSegEta;
class L1MuBMTrackFinder;
class L1MuBMTrack;
class L1MuBMTEtaPatternLut;
class L1MuBMTQualPatternLut;
class L1MuDTTFMasks;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuBMEtaProcessor {

  public:

    /// constructor
    L1MuBMEtaProcessor(const L1MuBMTrackFinder&, int id, edm::ConsumesCollector&& iC );

    /// destructor
    virtual ~L1MuBMEtaProcessor();

    /// return Eta Processor identifier (0-11)
    inline int id() const { return m_epid; }

    /// run the Eta Processor
    virtual void run(int bx, const edm::Event& e, const edm::EventSetup& c);

    /// reset the Eta Processor
    virtual void reset();

    /// print muon candidates found by the Eta Processor
    void print() const;

    /// return reference to barrel MTTF
    inline const L1MuBMTrackFinder& tf() const { return m_tf; }

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

    const L1MuBMTrackFinder&                  m_tf;
    int                                       m_epid;

    int                                       m_mask;

    int                                       m_eta[12];
    bool                                      m_fine[12];

    std::vector<int>                          m_foundPattern;
    int                                       m_pattern[12];

    int                                       m_address[12];
    L1MuBMTrack*                              m_TrackCand[12];
    L1MuBMTrack*                              m_TracKCand[12];
    std::vector<const L1MuBMTrackSegEta*>     m_tseta;

    //edm::ESHandle< L1MuDTEtaPatternLut >  theEtaPatternLUT;  // ETF look-up table
    //edm::ESHandle< L1MuDTQualPatternLut > theQualPatternLUT; // EMU look-up tables
    //edm::ESHandle< L1MuDTTFMasks >        msks;
    edm::ESHandle< L1TMuonBarrelParams > bmtfParamsHandle;
    L1MuDTTFMasks       msks;
    L1MuBMTEtaPatternLut   theEtaPatternLUT;  // ETF look-up table
    L1MuBMTQualPatternLut  theQualPatternLUT; // EMU look-up tables

    edm::EDGetTokenT<L1MuDTChambThContainer>  m_DTDigiToken;

};

#endif
