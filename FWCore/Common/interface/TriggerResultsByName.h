#ifndef FWCore_Common_TriggerResultsByName_h
#define FWCore_Common_TriggerResultsByName_h
// -*- C++ -*-
//
// Package:     FWCore/Common
// Class  :     TriggerResultsByName
// 
/**\class TriggerResultsByName TriggerResultsByName.h FWCore/Common/interface/TriggerResultsByName.h

 Description: Class which provides methods to access trigger results

 Usage:
    This class is intended to make it convenient and easy to
 get the trigger results from each event starting with the name
 of a trigger path.  One obtains an object of this class from
 the event using the Event::triggerResultsByName function.
 One can use this class for code which needs to work in both
 the full and the light (i.e. FWLite) frameworks.

    Once the user has an object of this class, the user can use
 the accessors below to get the trigger results for a trigger in
 one step, instead of first using the TriggerNames class to get
 the index and then using the index and the TriggerResults class
 to get the result.

    While the most common use of this class will be to get the
 results of the triggers from the HLT process, this class can
 also be used to get the results of the paths of any process
 used to create the input file.
*/
//
// Original Author:  W. David Dagenhart
//         Created:  11 December 2009
//

#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Common/interface/HLTenums.h"

#include <string>
#include <vector>

namespace edm {

  class TriggerResults;
  class TriggerNames;
  class HLTPathStatus;

  class TriggerResultsByName {

  public:

    TriggerResultsByName(TriggerResults const* triggerResults,
                         TriggerNames const* triggerNames);

    bool isValid() const;

    ParameterSetID const& parameterSetID() const;

    // Was at least one path run?
    bool wasrun() const;

    // Has at least one path accepted the event?
    bool accept() const;

    // Has any path encountered an error (exception)
    bool  error() const;

    HLTPathStatus const& at(std::string const& pathName) const;
    HLTPathStatus const& at(unsigned i) const;

    HLTPathStatus const& operator[](std::string const& pathName) const;
    HLTPathStatus const& operator[](unsigned i) const;

    // Was ith path run?
    bool wasrun(std::string const& pathName) const;
    bool wasrun(unsigned i) const;

    // Has ith path accepted the event
    bool accept(std::string const& pathName) const;
    bool accept(unsigned i) const;

    // Has ith path encountered an error (exception)?
    bool error(std::string const& pathName) const;
    bool error(unsigned i) const;

    // Get status of ith path
    hlt::HLTState state(std::string const& pathName) const;
    hlt::HLTState state(unsigned i) const;

    // Get index (slot position) of module giving the decision of the ith path
    unsigned index(std::string const& pathName) const;
    unsigned index(unsigned i) const;

    std::vector<std::string> const& triggerNames() const;

    // Throws if the number is out of range.
    std::string const& triggerName(unsigned i) const;

    // If the input name is not known, this returns a value
    // equal to the size.
    unsigned triggerIndex(std::string const& pathName) const;

    // The number of trigger names.
    std::vector<std::string>::size_type size() const;

  private:

    unsigned getAndCheckIndex(std::string const& pathName) const;

    void throwTriggerResultsMissing() const;
    void throwTriggerNamesMissing() const;

    TriggerResults const* triggerResults_;
    TriggerNames const* triggerNames_;
  };
}
#endif
