#ifndef Framework_TRIGGERNAMES_h
#define Framework_TRIGGERNAMES_h

// -*- C++ -*-
/*

 Original Author:  W. David Dagenhart 1 June 2007

 $Id$

 Used to access the names and indices of the triggers corresponding
 to a particular TriggerResults object.  Uses the TriggerNamesService
 to get those names.

 This class is smart enough to only update the names and index map
 when the ID stored in the TriggerResults object changes (and it lets
 you know when that happens).

 Should you use this class or the TriggerNamesService directly?  If
 you are interested in the names for the current process, it is
 better to use the service directly.  On the other hand for previous
 processes, the service will only return a vector with all the
 trigger names.  This class has accessors in that can return one
 name when given an index and vice versa. If you need those accessors
 then you might want to use this class.  The service won't do that
 for previous processes. But this class requires some additional
 memory and CPU overhead to build and store the trigger names and
 the map to indices (which is why they are not stored in the service
 to begin with, in some cases there might be many sets of trigger
 names).  The combination of this class and the service gives one
 options to optimize memory, CPU usage, and convenience for different
 use cases.

*/

#include "DataFormats/Provenance/interface/ParameterSetID.h"

#include <string>
#include <map>
#include <vector>

namespace edm {

  class TriggerResults;

  class TriggerNames {

  public:

    typedef std::vector<std::string> Strings;
    typedef std::map<std::string, unsigned int> IndexMap;

    TriggerNames();
    TriggerNames(TriggerResults const& triggerResults);

    // Returns true if the trigger names actually change.
    // If the ID stored in TriggerResults is the same
    // as the one from the previous call to init, then
    // the trigger names are also the same.  In this case,
    // this function immediately returns false and does not
    // waste any time or CPU cycles.
    bool init(TriggerResults const& triggerResults);

    Strings const& triggerNames() const;

    // Throws if the number is out of range.
    std::string const& triggerName(unsigned int index) const;

    // If the input name is not known, this returns a value
    // equal to the size.
    unsigned int triggerIndex(std::string const& name) const;

    // The number of trigger names.
    Strings::size_type size() const;

  private:

    bool psetID_valid_;
    ParameterSetID psetID_;

    Strings triggerNames_;
    IndexMap indexMap_;

  };
}

#endif
