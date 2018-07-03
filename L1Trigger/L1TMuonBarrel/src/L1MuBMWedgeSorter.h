//-------------------------------------------------
//
/**  \class L1MuBMWedgeSorter
 *
 *   Wedge Sorter:
 *
 *   A Wedge Sorter receives 2 muon candidates
 *   from each of the 6 Sector Processors of a
 *   wedge and forwards the 2 highest rank
 *   candidates per wedge to the BM Muon Sorter
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBM_WEDGE_SORTER_H
#define L1MUBM_WEDGE_SORTER_H

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

class L1MuBMWedgeSorter : public L1AbstractProcessor {

  public:

    /// constructor
    L1MuBMWedgeSorter(const L1MuBMTrackFinder&, int id );

    /// destructor
    ~L1MuBMWedgeSorter() override;

    /// return Wedge Sorter identifier (0-11)
    inline int id() const { return m_wsid; }

    /// run Wedge Sorter
    void run() override;

    /// reset Wedge Sorter
    void reset() override;

    /// print results after sorting
    void print() const;

    /// return vector of muon candidates
    inline const std::vector<const L1MuBMTrack*>& tracks() const { return m_TrackCands; }

    /// return number of muon candidates
    inline bool anyMuonCands() const { return anyTrack(); }

  private:

    /// run the Cancel Out Logic of the wedge sorter
    void runCOL(std::vector<L1MuBMTrack*>&) const;

    /// are there any non-empty muon candidates in the Wedge Sorter?
    bool anyTrack() const;

    /// find out if two Sector Processors are neighbours in the same wedge
    static bool neighbour(const L1MuBMSecProcId& spid1, const L1MuBMSecProcId& spid2);

  private:

    const L1MuBMTrackFinder&        m_tf;
    int                             m_wsid;

    std::vector<const L1MuBMTrack*> m_TrackCands;

};

#endif
