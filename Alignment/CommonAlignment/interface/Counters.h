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
 *
 *  $Date: 2007/04/09 00:40:21 $
 *  $Revision: 1.7 $
 *  \author Chung Khim Lae
 */

#include <map>

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "CondFormats/Alignment/interface/Definitions.h"

namespace align
{
  typedef unsigned int (*Counter)(align::ID);
}

class Counters
{
  public:

  /// Get a counter based on its structure type.
  static align::Counter get(
			    align::StructureType
			    );

  private:

  /// Build the counters map.
  Counters();

  static std::map<align::StructureType, align::Counter> theCounters;
};

#endif
