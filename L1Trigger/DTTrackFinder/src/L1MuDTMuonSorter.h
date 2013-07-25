//-------------------------------------------------
//
/**  \class L1MuDTMuonSorter
 *
 *   DT Muon Sorter:
 *
 *   The DT Muon Sorter receives 2 muon
 *   candidates from each of the 
 *   12 Wedge Sorters and sorts out the
 *   4 best (highest pt, highest quality) muons
 *
 *
 *   $Date: 2007/03/30 09:05:32 $
 *   $Revision: 1.3 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_MUON_SORTER_H
#define L1MUDT_MUON_SORTER_H

//---------------
// C++ Headers --
//---------------

#include <vector>

//----------------------
// Base Class Headers --
//----------------------

#include "L1Trigger/DTTrackFinder/interface/L1AbstractProcessor.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

class L1MuDTTrackFinder;
class L1MuDTTrack;
class L1MuDTSecProcId;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTMuonSorter : public L1AbstractProcessor {

  public:

    /// constructor
    L1MuDTMuonSorter(const L1MuDTTrackFinder&);

    /// destructor
    virtual ~L1MuDTMuonSorter();

    /// run Muon Sorter
    virtual void run();
    
    /// reset Muon Sorter
    virtual void reset();
    
    /// print results after sorting
    void print() const;

    /// return number of found muon candidates after sorter
    inline int numberOfTracks() const { return m_TrackCands.size(); }
    
    /// return pointer to a muon candidate
    inline const L1MuDTTrack* track(int id) const { return m_TrackCands[id]; }
    
    /// return vector of muon candidates
    inline const std::vector<const L1MuDTTrack*>& tracks() const { return m_TrackCands; }

  private:

    /// run the Cancel Out Logic of the muon sorter
    void runCOL(std::vector<L1MuDTTrack*>&) const; 

    /// find out if two Sector Processors are neighbours
    static int neighbour(const L1MuDTSecProcId& spid1, const L1MuDTSecProcId& spid2);

  private:

    const L1MuDTTrackFinder&        m_tf;
    std::vector<const L1MuDTTrack*> m_TrackCands;

};

#endif
