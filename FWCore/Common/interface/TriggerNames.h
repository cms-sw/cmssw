#ifndef FWCore_Common_TriggerNames_h
#define FWCore_Common_TriggerNames_h

// -*- C++ -*-
/*
 Original Author:  W. David Dagenhart 1 June 2007
 (originally this was in the FWCore/Framework package)

 Used to access the names and indices of the triggers corresponding
 to a particular TriggerResults object (usually from the HLT process
 but one could also access the path names of other cmsRun processes
 used to create the input file).

 A user should get a reference to an instance of this class using
 the triggerNames function in the Event class (works in both FWLite
 and the full Framework).  Make sure you assign the value returned
 from that function to a reference or else you will deep copy the
 object and waste both CPU and memory.  The Event maintains a cache
 of TriggerNames objects for rapid lookup. There should be no reason
 for each module that uses TriggerNames to maintain its own cache
 of TriggerNames objects, not even of the current object.

 Some users may need to optimize the performance of their code and
 do most of their work using the integer indices instead of repeatedly
 working with the strings used to store the names. Often this will
 require initializing some data structures and this initialization
 will be more efficient if only done when the names actually change.
 One can quickly test if the names are the same by saving the previous
 ParameterSetID. Then for each event you compare the ParameterSetID
 for the current names to the previous one, and only reinitialize
 if the ID changes instead of reinitializing on every event. (Although
 generally for real data we expect the names will only change at run
 boundaries, there already exist datasets where they change more often
 than that in simulation. There might be some strange cases where
 they also change in real data.  There is nothing in the offline
 code to prevent that)

 When using the ParameterSetID, one should also check that the ParameterSetID
 is valid (the default constructed one is the invalid value, ParameterSetID()).
 For very old format data, the names are stored in TriggerResults itself and
 in that case the ParameterSetID will always be invalid and it tells you nothing
 about whether or not the names changed.
*/

#include "DataFormats/Provenance/interface/ParameterSetID.h"

#include <string>
#include <map>
#include <vector>

namespace edm {

  class TriggerResults;
  class ParameterSet;

  class TriggerNames {

  public:

    typedef std::vector<std::string> Strings;
    typedef std::map<std::string, unsigned int> IndexMap;

    // Users should not construct these.  Instead they should
    // get a reference to the current one from the Event. See
    // comments above.

    TriggerNames();

    TriggerNames(edm::ParameterSet const& pset);

    Strings const& triggerNames() const;

    // Throws if the number is out of range.
    std::string const& triggerName(unsigned int index) const;

    // If the input name is not known, this returns a value
    // equal to the size.
    unsigned int triggerIndex(std::string const& name) const;

    // The number of trigger names.
    Strings::size_type size() const;

    // Can be used to quickly compare two TriggerNames objects
    // to see whether or not they contain the same names.
    ParameterSetID const& parameterSetID() const;

    // The next constructor and the init function that
    // follows are deprecated.  We keep them for now for
    // backward compatibility reasons, but they may
    // get deleted one of these days.  They will not
    // work properly in FWLite unless the client code
    // explicitly forces the ParameterSet registry to be
    // read from the input file before calling them

    TriggerNames(TriggerResults const& triggerResults);

    // Returns true if the trigger names actually change.
    // If the ID stored in TriggerResults is the same
    // as the one from the previous call to init, then
    // the trigger names are also the same.  In this case,
    // this function immediately returns false and does not
    // waste any time or CPU cycles.
    bool init(TriggerResults const& triggerResults);

  private:

    ParameterSetID psetID_;

    Strings triggerNames_;
    IndexMap indexMap_;

  };
}

#endif
