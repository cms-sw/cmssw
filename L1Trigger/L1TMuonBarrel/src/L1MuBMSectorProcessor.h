//-------------------------------------------------
//
/**  \class L1MuBMSectorProcessor
 *
 *   Sector Processor:
 *
 *   A Sector Processor consists of:
 *    - one Data Buffer,
 *    - one Extrapolation Unit (EU),
 *    - one Track Assembler (TA) and
 *    - two Assignment Units (AU)
 *
 *
 *
 *   N. Neumeister            CERN EP
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef L1MUBM_SECTOR_PROCESSOR_H
#define L1MUBM_SECTOR_PROCESSOR_H

//---------------
// C++ Headers --
//---------------

#include <vector>

//----------------------
// Base Class Headers --
//----------------------

#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMSecProcId.h"
#include "DataFormats/L1TMuon/interface/L1MuBMTrack.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
class L1MuBMSectorReceiver;
class L1MuBMDataBuffer;
class L1MuBMExtrapolationUnit;
class L1MuBMTrackAssembler;
class L1MuBMAssignmentUnit;
class L1MuBMTrackFinder;
class L1MuBMTrack;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuBMSectorProcessor {

  public:

    /// constructor
    L1MuBMSectorProcessor(const L1MuBMTrackFinder&, const L1MuBMSecProcId&, edm::ConsumesCollector&&);

    /// destructor
    virtual ~L1MuBMSectorProcessor();

    /// run the Sector Processor
    virtual void run(int bx, const edm::Event& e, const edm::EventSetup& c);

    /// reset the Sector Processor
    virtual void reset();

    /// print muon candidates found by the Sector Processor
    void print() const;

    /// return pointer to the next wheel neighbour
    const L1MuBMSectorProcessor* neighbour() const;

    /// return Sector Processor identifier
    inline const L1MuBMSecProcId& id() const { return m_spid; }

    /// return reference to barrel MTTF
    inline const L1MuBMTrackFinder& tf() const { return m_tf; }

    /// is it a barrel-only Sector Processor?
    inline bool brl() const { return !m_spid.ovl(); }

    /// is it an overlap region Sector Processor?
    inline bool ovl() const { return m_spid.ovl(); }

    /// return pointer to Data Buffer
    inline const L1MuBMDataBuffer* data() const { return m_DataBuffer; }
    inline L1MuBMDataBuffer* data() { return m_DataBuffer; }

    /// return pointer to Extrapolation Unit
    inline const L1MuBMExtrapolationUnit* EU() const { return m_EU; }

    /// return pointer to Track Assembler
    inline const L1MuBMTrackAssembler* TA() const { return m_TA; }

    /// return pointer to Assignment Unit, index [0,1]
    inline const L1MuBMAssignmentUnit* AU(int id) const { return m_AUs[id]; }

    /// return pointer to muon candidate, index [0,1]
    inline L1MuBMTrack* track(int id) const { return m_TrackCands[id]; }

    /// return pointer to muon candidate, index [0,1]
    inline L1MuBMTrack* tracK(int id) const { return m_TracKCands[id]; }

  private:

    /// are there any non-empty muon candidates?
    bool anyTrack() const;

  private:

    const L1MuBMTrackFinder&            m_tf;
    L1MuBMSecProcId                     m_spid;

    L1MuBMSectorReceiver*               m_SectorReceiver;
    L1MuBMDataBuffer*                   m_DataBuffer;
    L1MuBMExtrapolationUnit*            m_EU;
    L1MuBMTrackAssembler*               m_TA;
    std::vector<L1MuBMAssignmentUnit*>  m_AUs;

    std::vector<L1MuBMTrack*>           m_TrackCands;
    std::vector<L1MuBMTrack*>           m_TracKCands;

};

#endif
