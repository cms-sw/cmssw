//-------------------------------------------------
//
/**  \class L1MuBMSecProcMap
 *
 *   Sector Processor container
 *   map which contains all sector processors
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBM_SEC_PROC_MAP_H
#define L1MUBM_SEC_PROC_MAP_H

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

#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMSecProcId.h"

class L1MuBMSectorProcessor;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuBMSecProcMap {

  public:

    typedef std::map<L1MuBMSecProcId, L1MuBMSectorProcessor*, std::less<L1MuBMSecProcId> >  SPmap;
    typedef SPmap::iterator                                   SPmap_iter;

    /// constructor
    L1MuBMSecProcMap();

    /// destructor
    virtual ~L1MuBMSecProcMap();

    /// return pointer to Sector Processor
    L1MuBMSectorProcessor* sp(const L1MuBMSecProcId& ) const;

    /// insert a Sector Processor into the container
    void insert(const L1MuBMSecProcId&, L1MuBMSectorProcessor* sp);

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
