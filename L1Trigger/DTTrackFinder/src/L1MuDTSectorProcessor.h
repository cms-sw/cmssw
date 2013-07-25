//-------------------------------------------------
//
/**  \class L1MuDTSectorProcessor
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
 *   $Date: 2008/05/09 15:01:59 $
 *   $Revision: 1.4 $
 *
 *   N. Neumeister            CERN EP
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef L1MUDT_SECTOR_PROCESSOR_H
#define L1MUDT_SECTOR_PROCESSOR_H

//---------------
// C++ Headers --
//---------------

#include <vector>

//----------------------
// Base Class Headers --
//----------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTSecProcId.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include <FWCore/Framework/interface/Event.h>
class L1MuDTSectorReceiver;
class L1MuDTDataBuffer;
class L1MuDTExtrapolationUnit;
class L1MuDTTrackAssembler;
class L1MuDTAssignmentUnit;
class L1MuDTTrackFinder;
class L1MuDTTrack;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTSectorProcessor {

  public:

    /// constructor
    L1MuDTSectorProcessor(const L1MuDTTrackFinder&, const L1MuDTSecProcId& );

    /// destructor
    virtual ~L1MuDTSectorProcessor();

    /// run the Sector Processor
    virtual void run(int bx, const edm::Event& e, const edm::EventSetup& c);

    /// reset the Sector Processor
    virtual void reset();

    /// print muon candidates found by the Sector Processor
    void print() const;

    /// return pointer to the next wheel neighbour
    const L1MuDTSectorProcessor* neighbour() const;
    
    /// return Sector Processor identifier
    inline const L1MuDTSecProcId& id() const { return m_spid; }
    
    /// return reference to barrel MTTF
    inline const L1MuDTTrackFinder& tf() const { return m_tf; }

    /// is it a barrel-only Sector Processor?
    inline bool brl() const { return !m_spid.ovl(); }

    /// is it an overlap region Sector Processor? 
    inline bool ovl() const { return m_spid.ovl(); }

    /// return pointer to Data Buffer
    inline const L1MuDTDataBuffer* data() const { return m_DataBuffer; }
    inline L1MuDTDataBuffer* data() { return m_DataBuffer; }
    
    /// return pointer to Extrapolation Unit
    inline const L1MuDTExtrapolationUnit* EU() const { return m_EU; }

    /// return pointer to Track Assembler
    inline const L1MuDTTrackAssembler* TA() const { return m_TA; }

    /// return pointer to Assignment Unit, index [0,1]
    inline const L1MuDTAssignmentUnit* AU(int id) const { return m_AUs[id]; }

    /// return pointer to muon candidate, index [0,1]
    inline L1MuDTTrack* track(int id) const { return m_TrackCands[id]; }

    /// return pointer to muon candidate, index [0,1]
    inline L1MuDTTrack* tracK(int id) const { return m_TracKCands[id]; }
    
  private:

    /// are there any non-empty muon candidates?
    bool anyTrack() const;

  private:

    const L1MuDTTrackFinder&            m_tf;
    L1MuDTSecProcId                     m_spid;

    L1MuDTSectorReceiver*               m_SectorReceiver;
    L1MuDTDataBuffer*                   m_DataBuffer;
    L1MuDTExtrapolationUnit*            m_EU;
    L1MuDTTrackAssembler*               m_TA;
    std::vector<L1MuDTAssignmentUnit*>  m_AUs;

    std::vector<L1MuDTTrack*>           m_TrackCands;
    std::vector<L1MuDTTrack*>           m_TracKCands;
 
};

#endif
