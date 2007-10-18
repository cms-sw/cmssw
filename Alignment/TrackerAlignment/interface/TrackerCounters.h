#ifndef Alignment_TrackerAlignment_TrackerCounters_H
#define Alignment_TrackerAlignment_TrackerCounters_H

/** \class TrackerCounters
 *
 *  Concrete implementation of counters for the tracker
 *
 *  Allows to set an id to each alignable. 
 *  Actual counter definitions are in separate header files.
 *  
 *  $Date: 2007/10/08 13:36:11 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */

#include <map>

#include "Alignment/CommonAlignment/interface/Counters.h"

class TrackerCounters : public Counters
{

public:
  /// Build the counters map.
  TrackerCounters();

  ~TrackerCounters() {}

};

#endif
