//-------------------------------------------------
//
/**  \class L1MuDTSecProcMap
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
#ifndef L1MUDT_SEC_PROC_MAP_H
#define L1MUDT_SEC_PROC_MAP_H

//---------------
// C++ Headers --
//---------------

#include <map>
#include <memory>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "L1Trigger/DTTrackFinder/interface/L1MuDTSecProcId.h"
class L1MuDTSectorProcessor;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTSecProcMap {
public:
  typedef std::map<L1MuDTSecProcId, std::unique_ptr<L1MuDTSectorProcessor>> SPmap;
  typedef SPmap::iterator SPmap_iter;

  /// constructor
  L1MuDTSecProcMap();

  /// destructor
  ~L1MuDTSecProcMap();

  /// return pointer to Sector Processor
  const L1MuDTSectorProcessor* sp(const L1MuDTSecProcId&) const;

  /// insert a Sector Processor into the container
  void insert(const L1MuDTSecProcId&, std::unique_ptr<L1MuDTSectorProcessor> sp);

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
