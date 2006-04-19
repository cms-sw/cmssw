#ifndef Services_TRIGGERNAMES_h
#define Services_TRIGGERNAMES_h

// -*- C++ -*-
/*

 Original Author:  Jim Kowalkowski 26-01-06

 $Id: TriggerNamesService.h,v 1.4 2006/04/10 22:35:43 jbk Exp $

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
    using std::string;
    class TriggerNamesService
    {
    public:
      typedef std::vector<std::string> Strings;
      typedef std::vector<bool> Bools;
      typedef std::map<std::string, unsigned int> PosMap;

      TriggerNamesService(const ParameterSet& proc_pset);
      ~TriggerNamesService();

      // trigger path information after all configuration options are applied
      Bools getBitMask(const Strings& interesting_names) const;

      // info from configuration script
      std::string getProcessName() const { return process_name_; }

      const Strings& getEndPaths() const { return end_names_; }
      const string&  getEndPath(const unsigned int i) const { return end_names_.at(i);}
      unsigned int  findEndPath(const string& name) const { return find(end_pos_,name);}

      const Strings& getPaths() const { return pathnames_; }
      const string&  getPath(const unsigned int i) const { return pathnames_.at(i);}
      unsigned int  findPath(const string& name) const { return find(pathpos_,name);}

      const Strings& getTrigPaths() const { return trignames_; }
      const string&  getTrigPath(const unsigned int i) const { return trignames_.at(i);}
      unsigned int  findTrigPath(const string& name) const { return find(trigpos_,name);}

      const Strings& getTrigPathModules(const string& name) const {
	return modulenames_.at(find(trigpos_,name));
      }
      const Strings& getTrigPathModules(const unsigned int i) const {
	return modulenames_.at(i);
      }
      const string&  getTrigPathModule (const string& name, const unsigned int j) const {
	return (modulenames_.at(find(trigpos_,name))).at(j);
      }
      const string&  getTrigPathModule (const unsigned int i, const unsigned int j) const {
	return (modulenames_.at(i)).at(j);
      }

      unsigned int find (const PosMap& posmap, const string& name) const {
	const PosMap::const_iterator pos(posmap.find(name));
        if (pos==posmap.end()) {
	  return posmap.size();
	} else {
          return pos->second;
	}
      }

      void loadPosMap(PosMap& posmap, const Strings& names) {
        const unsigned int n(names.size());
	for (unsigned int i=0; i!=n; i++) {
	  posmap[names[i]] = i;
	}
      }
      
      bool wantSummary() const { return wantSummary_; }
      bool makeTriggerResults() const { return makeTriggerResults_; }
      edm::ParameterSet getTrigPSet() const { return trig_pset_; }

    private:
      std::string process_name_;
      edm::ParameterSet trig_pset_;

      Strings trignames_; // of trigger paths
      PosMap  trigpos_;
      Strings pathnames_;
      PosMap  pathpos_;
      Strings end_names_;
      PosMap  end_pos_;

      std::vector<Strings> modulenames_; // of modules on trigger paths

      bool wantSummary_;
      bool makeTriggerResults_;
    };
  }
}



#endif
