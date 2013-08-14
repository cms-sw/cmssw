//-------------------------------------------------
//
/**  \class L1MuDTWedgeSorter
 *
 *   Wedge Sorter:
 *
 *   A Wedge Sorter receives 2 muon candidates
 *   from each of the 6 Sector Processors of a 
 *   wedge and forwards the 2 highest rank 
 *   candidates per wedge to the DT Muon Sorter
 *
 *
 *   $Date: 2007/03/30 09:05:32 $
 *   $Revision: 1.3 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_WEDGE_SORTER_H
#define L1MUDT_WEDGE_SORTER_H

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

class L1MuDTWedgeSorter : public L1AbstractProcessor {

  public:

    /// constructor
    L1MuDTWedgeSorter(const L1MuDTTrackFinder&, int id );

    /// destructor
    virtual ~L1MuDTWedgeSorter();
 
    /// return Wedge Sorter identifier (0-11)
    inline int id() const { return m_wsid; }
    
    /// run Wedge Sorter
    virtual void run();
    
    /// reset Wedge Sorter
    virtual void reset();
    
    /// print results after sorting 
    void print() const;

    /// return vector of muon candidates
    inline const std::vector<const L1MuDTTrack*>& tracks() const { return m_TrackCands; }
  
  private:

    /// run the Cancel Out Logic of the wedge sorter
    void runCOL(std::vector<L1MuDTTrack*>&) const; 
    
    /// are there any non-empty muon candidates in the Wedge Sorter?
    bool anyTrack() const;
    
    /// find out if two Sector Processors are neighbours in the same wedge
    static bool neighbour(const L1MuDTSecProcId& spid1, const L1MuDTSecProcId& spid2);

  private:

    const L1MuDTTrackFinder&        m_tf;
    int                             m_wsid;

    std::vector<const L1MuDTTrack*> m_TrackCands;

};

#endif
