#ifndef DataFormats_FWLite_TriggerNames_h
#define DataFormats_FWLite_TriggerNames_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     TriggerNames
// 
/**\class TriggerNames TriggerNames.h DataFormats/FWLite/interface/TriggerNames.h

   Description: In an FWLite process, one can use this class to 
access the path names from the cmsRun processes used to create
the input file.  People will be most interested in the HLT process,
and the HLT path names are also known as the trigger names.  One
normally obtains a fully initialized TriggerNames object from a
class that inherits from the fwlite::EventBase class.

This class has the same accessors as the edm::TriggerNames class
which is used in the full Framework.  The names returned by the
accessors should be the same.  The methods used to set the values
in fwlite::TriggerNames and edm::TriggerNames objects are different.
*/
//
// Original Author:  W. David Dagenhart
//         Created:  28 August 2009
//

#include <vector>
#include <string>
#include <map>

namespace edm {
   class ParameterSet;
}

namespace fwlite
{
   class TriggerNames {
   public:

      typedef std::vector<std::string> Strings;
      typedef std::map<std::string, unsigned int> IndexMap;

      TriggerNames(edm::ParameterSet const& pset);

      ~TriggerNames();

      Strings const& triggerNames() const;

      // Throws if the number is out of range.
      std::string const& triggerName(unsigned int index) const;

      // If the input name is not known, this returns a value
      // equal to the size.
      unsigned int triggerIndex(std::string const& name) const;

      // The number of trigger names.
      Strings::size_type size() const;

   private:

      Strings triggerNames_;
      IndexMap indexMap_;
   };
}
#endif
