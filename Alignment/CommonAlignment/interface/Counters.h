#ifndef Alignment_CommonAlignment_Counters_H
#define Alignment_CommonAlignment_Counters_H

/** \class Counters
 *
 *  Class to store a list of counters.
 *
 *  A counter is a pointer to a function that returns the number of an
 *  alignable based on its id.
 *  The number of an alignable is given by its position within its parent.
 *  User gets a counter using its structure type via Counters::get(type).
 *  Each sub-system has its own concrete counter class implementation.
 *  
 *  $Date: 2007/10/18 09:41:07 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

#include <map>

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "CondFormats/Alignment/interface/Definitions.h"

class TrackerTopology;
namespace align
{
  typedef unsigned int (*Counter)(align::ID, const TrackerTopology*);
}

class Counters
{
  public:

  /// Build the counters map.
  Counters() {}

  virtual ~Counters() {}

  /// Get a counter based on its structure type.
  virtual align::Counter get( align::StructureType ) const;

protected:
  std::map<align::StructureType, align::Counter> theCounters;

};

#endif
