#ifndef Alignment_CommonAlignment_Counters_H
#define Alignment_CommonAlignment_Counters_H

/** \class AlignableIndexer
 *
 *  Class to store a list of index functions.
 *
 *  A counter is a pointer to a function that returns the number of an
 *  alignable based on its id.
 *  The number of an alignable is given by its position within its parent.
 *  User gets a counter using its structure type via AlignableIndexer::get(type).
 *  Each sub-system has its own concrete counter class implementation.
 *  
 *  $Date: 2007/10/18 09:41:07 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 *
 *  Last Update: Max Stark
 *         Date: Wed, 17 Feb 2016 15:39:06 CET
 */

#include <map>
#include <functional>

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "CondFormats/Alignment/interface/Definitions.h"

class TrackerTopology;
namespace align
{
  using Counter = std::function<unsigned int(align::ID)>;
}

class AlignableIndexer
{
  public:

  /// Build the counters map.
  AlignableIndexer() = default;
  AlignableIndexer(const AlignableIndexer&) = default;
  AlignableIndexer& operator=(const AlignableIndexer&) = default;
  AlignableIndexer(AlignableIndexer&&) = default;
  AlignableIndexer& operator=(AlignableIndexer&&) = default;
  virtual ~AlignableIndexer() = default;

  /// Get a counter based on its structure type.
  virtual align::Counter get(align::StructureType, const AlignableObjectId&) const;

protected:
  std::map<align::StructureType, align::Counter> theCounters;

};

#endif
