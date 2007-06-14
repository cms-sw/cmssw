#ifndef FWCore_Framework_TriggerNamesService_h
#define FWCore_Framework_TriggerNamesService_h

// -*- C++ -*-
/*

 Original Author:  Jim Kowalkowski 26-01-06

 $Id: TriggerNamesService.h,v 1.6 2007/01/23 00:32:02 wmtan Exp $

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

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  namespace service {
    class TriggerNamesService
    {
    public:
      typedef std::vector<std::string> Strings;
      typedef std::vector<bool> Bools;
      typedef std::map<std::string, unsigned int> PosMap;

      TriggerNamesService(ParameterSet const& proc_pset);
      ~TriggerNamesService();

      // trigger path information after all configuration options are applied
      Bools getBitMask(Strings const& interesting_names) const;

      // info from configuration script
      std::string getProcessName() const { return process_name_; }

      Strings const& getEndPaths() const { return end_names_; }
      std::string const&  getEndPath(unsigned int const i) const { return end_names_.at(i);}
      unsigned int  findEndPath(std::string const& name) const { return find(end_pos_,name);}

      Strings const& getPaths() const { return pathnames_; }
      std::string const&  getPath(unsigned int const i) const { return pathnames_.at(i);}
      unsigned int  findPath(std::string const& name) const { return find(pathpos_,name);}

      Strings const& getTrigPaths() const { return trignames_; }
      std::string const&  getTrigPath(unsigned int const i) const { return trignames_.at(i);}
      unsigned int  findTrigPath(std::string const& name) const { return find(trigpos_,name);}

      Strings const& getTrigPathModules(std::string const& name) const {
	return modulenames_.at(find(trigpos_,name));
      }
      Strings const& getTrigPathModules(unsigned int const i) const {
	return modulenames_.at(i);
      }
      std::string const&  getTrigPathModule (std::string const& name, unsigned int const j) const {
	return (modulenames_.at(find(trigpos_,name))).at(j);
      }
      std::string const&  getTrigPathModule (unsigned int const i, unsigned int const j) const {
	return (modulenames_.at(i)).at(j);
      }

      unsigned int find (PosMap const& posmap, std::string const& name) const {
	PosMap::const_iterator const pos(posmap.find(name));
        if (pos == posmap.end()) {
	  return posmap.size();
	} else {
          return pos->second;
	}
      }

      void loadPosMap(PosMap& posmap, Strings const& names) {
        unsigned int const n(names.size());
	for (unsigned int i = 0; i != n; ++i) {
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
