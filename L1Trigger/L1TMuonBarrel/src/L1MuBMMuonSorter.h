//-------------------------------------------------
//
/**  \class L1MuBMMuonSorter
 *
 *   BM Muon Sorter:
 *
 *   The BM Muon Sorter receives 2 muon
 *   candidates from each of the
 *   12 Wedge Sorters and sorts out the
 *   4 best (highest pt, highest quality) muons
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBM_MUON_SORTER_H
#define L1MUBM_MUON_SORTER_H

//---------------
// C++ Headers --
//---------------

#include <vector>

//----------------------
// Base Class Headers --
//----------------------

#include "L1Trigger/L1TMuonBarrel/interface/L1AbstractProcessor.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

class L1MuBMTrackFinder;
class L1MuBMTrack;
class L1MuBMSecProcId;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuBMMuonSorter : public L1AbstractProcessor {

  public:

    /// constructor
    L1MuBMMuonSorter(const L1MuBMTrackFinder&);

    /// destructor
    ~L1MuBMMuonSorter() override;

    /// run Muon Sorter
    void run() override;

    /// reset Muon Sorter
    void reset() override;

    /// print results after sorting
    void print() const;

    /// return number of found muon candidates after sorter
    inline int numberOfTracks() const { return m_TrackCands.size(); }

    /// return pointer to a muon candidate
    inline const L1MuBMTrack* track(int id) const { return m_TrackCands[id]; }

    /// return vector of muon candidates
    inline const std::vector<const L1MuBMTrack*>& tracks() const { return m_TrackCands; }

  private:

    /// run the Cancel Out Logic of the muon sorter
    void runCOL(std::vector<L1MuBMTrack*>&) const;

    /// find out if two Sector Processors are neighbours
    static int neighbour(const L1MuBMSecProcId& spid1, const L1MuBMSecProcId& spid2);

  private:

    const L1MuBMTrackFinder&        m_tf;
    std::vector<const L1MuBMTrack*> m_TrackCands;

};

#endif
