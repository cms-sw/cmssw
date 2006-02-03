#ifndef Services_TRIGGERNAMES_h
#define Services_TRIGGERNAMES_h

// -*- C++ -*-
/*

 Original Author:  Jim Kowalkowski 26-01-06

 $Id: TriggerNamesService.h,v 1.1 2006/01/29 23:33:57 jbk Exp $

 This service makes the trigger bit assignments for the current process
 available to all modules.  This of particular use in the output modules.
 Given a sequence of trigger path names, this object will translate them
 into a vector<bool> that can be used in a comparison with a TriggerResults
 object.

 Note: This implementation does not make trigger bit assignments available
 from previous "process_names", such as HLT, if we are running production.

*/

#include <string>
#include <map>
#include <vector>

#include "FWCore/Framework/interface/TriggerResults.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  namespace service {
    class TriggerNamesService
    {
    public:
      typedef std::vector<std::string> Strings;
      typedef edm::TriggerResults::BitVector BitVector;
      typedef std::map<std::string, int> PosMap;

      TriggerNamesService(const ParameterSet& trigger_names,
			  const std::string& process_name);
      ~TriggerNamesService();
      
      BitVector getBitMask(const Strings& interesting_names) const;
      void getNames(Strings& out) const { out = names_; }
      std::string getProcessName() const { return process_name_; }

    private:
      std::string process_name_;
      Strings names_;
      PosMap pos_;
    };
  }
}



#endif
