#ifndef Alignment_TrackerAlignment_TrackerCounters_H
#define Alignment_TrackerAlignment_TrackerCounters_H

/** \class TrackerAlignableIndexer
 *
 *  Concrete implementation of AlignableIndexer for the tracker-alignables.
 *
 *  Allows to set an id to each alignable.
 *  Actual counter definitions are in separate header files.
 *
 *  $Date: 2007/10/08 13:36:11 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 *
 *  Last Update: Max Stark
 *         Date: Wed, 17 Feb 2016 15:39:06 CET
 */

#include <map>

#include "Alignment/CommonAlignment/interface/AlignableIndexer.h"
#include "Alignment/TrackerAlignment/interface/TrackerNameSpace.h"

class TrackerAlignableIndexer : public AlignableIndexer {
public:
  /// Build the counters map.
  TrackerAlignableIndexer(const align::TrackerNameSpace&);
  TrackerAlignableIndexer(const TrackerAlignableIndexer&) = default;
  TrackerAlignableIndexer& operator=(const TrackerAlignableIndexer&) = delete;
  TrackerAlignableIndexer(TrackerAlignableIndexer&&) = default;
  TrackerAlignableIndexer& operator=(TrackerAlignableIndexer&&) = delete;
  ~TrackerAlignableIndexer() override = default;

private:
  const align::TrackerNameSpace tns_;
};

#endif
