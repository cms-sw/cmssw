//-------------------------------------------------
//
/**  \class L1MuDTSecProcMap
 *
 *   Sector Processor container
 *   map which contains all sector processors
 *
 *
 *   $Date: 2007/02/27 11:44:00 $
 *   $Revision: 1.2 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_SEC_PROC_MAP_H
#define L1MUDT_SEC_PROC_MAP_H

//---------------
// C++ Headers --
//---------------

#include <map>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTSecProcId.h"
class L1MuDTSectorProcessor;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTSecProcMap {

  public:

    typedef std::map<L1MuDTSecProcId, L1MuDTSectorProcessor*, std::less<L1MuDTSecProcId> >  SPmap;
    typedef SPmap::iterator                                   SPmap_iter;

    /// constructor
    L1MuDTSecProcMap();

    /// destructor
    virtual ~L1MuDTSecProcMap();

    /// return pointer to Sector Processor
    L1MuDTSectorProcessor* sp(const L1MuDTSecProcId& ) const;

    /// insert a Sector Processor into the container
    void insert(const L1MuDTSecProcId&, L1MuDTSectorProcessor* sp);
  
    /// return number of entries present in the container
    inline int size() const { return m_map.size(); }
  
    /// return iterator which points to the first entry of the container
    inline SPmap_iter begin() { return m_map.begin(); }
  
    /// return iterator which points to the one-past-last entry of the container
    inline SPmap_iter end() { return m_map.end(); }

  private:

    SPmap m_map;

};

#endif
