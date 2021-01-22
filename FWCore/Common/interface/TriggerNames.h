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

One normally gets a TriggerNames object from the event with a line
of code that looks like this:

const edm::TriggerNames & triggerNames = event.triggerNames(*triggerResults);

where "event" has type edm::Event and "triggerResults" has type
edm::Handle<edm::TriggerResults>. It is a very good idea to
check if the Handle is valid before dereferencing it. "*triggerResults"
will seg fault if the Handle is invalid.  Note the return value is
a reference. Also the accessors inside TriggerNames that return the
names return references. Code will perform faster and use less memory
if the references are used instead of needlessly copying the strings.
Note that the Event maintains a cache of TriggerNames objects for
rapid lookup. There should be no reason for each module that uses
TriggerNames to maintain its own cache of TriggerNames objects
or strings, not even of the current trigger names.

Some users may need to know when the trigger names have changed,
because they initialize data structures or book histograms or
something when this occurs.  This can be determined quickly and
efficiently by saving the ParameterSetID associated with a TriggerNames
object and then comparing with the ParameterSetID of subsequent objects.
If the ParameterSetIDs are the same, then all the names are the
same. This is much more efficient than comparing all the names
in the object. Although generally for real data we expect the names
should only change at run boundaries, there already exist simulation
samples where the names change more often than that. There is nothing
in the offline code to prevent this and it probably makes sense to check
for names changing more often than by run even in real data.
*/

#include "DataFormats/Provenance/interface/ParameterSetID.h"

#include <string>
#include <utility>
#include <vector>

namespace edm {

  class ParameterSet;

  class TriggerNames {
  public:
    using IndexMap = std::vector<std::pair<std::string_view, unsigned int>>;
    using Strings = std::vector<std::string>;

    // Users should not construct these.  Instead they should
    // get a reference to the current one from the Event. See
    // comments above.
    TriggerNames() = default;
    explicit TriggerNames(edm::ParameterSet const& pset);

    TriggerNames(TriggerNames const&);
    TriggerNames(TriggerNames&&) = default;
    TriggerNames& operator=(TriggerNames const&);
    TriggerNames& operator=(TriggerNames&&) = default;

    // called as part of reading back object from ROOT storage
    void initializeTriggerIndex();

    Strings const& triggerNames() const;

    // Throws if the index is out of range.
    std::string const& triggerName(unsigned int index) const;

    // If the input name is not known, this returns a value
    // equal to the size.
    unsigned int triggerIndex(std::string_view name) const;

    // The number of trigger names.
    std::size_t size() const;

    // Can be used to quickly compare two TriggerNames objects
    // to see whether or not they contain the same names.
    ParameterSetID const& parameterSetID() const;

  private:
    ParameterSetID psetID_;

    Strings triggerNames_;
    IndexMap indexMap_;
  };
}  // namespace edm
#endif
