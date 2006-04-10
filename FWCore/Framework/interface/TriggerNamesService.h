#ifndef Services_TRIGGERNAMES_h
#define Services_TRIGGERNAMES_h

// -*- C++ -*-
/*

 Original Author:  Jim Kowalkowski 26-01-06

 $Id: TriggerNamesService.h,v 1.3 2006/02/08 00:44:24 wmtan Exp $

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

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  namespace service {
    class TriggerNamesService
    {
    public:
      typedef std::vector<std::string> Strings;
      typedef edm::TriggerResults::BitVector BitVector;
      typedef std::map<std::string, int> PosMap;

      TriggerNamesService(const ParameterSet& proc_pset);
      ~TriggerNamesService();
      
      // trigger path information after all configuration options are applied
      BitVector getBitMask(const Strings& interesting_names) const;
      void getNames(Strings& out) const { out = names_; }

      // info from configuration script
      std::string getProcessName() const { return process_name_; }
      const Strings& getEndPaths() const { return end_paths_; }
      const Strings& getPaths() const { return paths_; }
      const Strings& getTrigPaths() const { return names_; }
      bool wantSummary() const { return wantSummary_; }
      bool makeTriggerResults() const { return makeTriggerResults_; }
      edm::ParameterSet getTrigPSet() const { return trig_pset_; }

    private:
      std::string process_name_;
      edm::ParameterSet trig_pset_;
      Strings names_; // of trigger paths
      Strings paths_;
      Strings end_paths_;
      bool wantSummary_;
      bool makeTriggerResults_;
      PosMap pos_;
    };
  }
}



#endif
