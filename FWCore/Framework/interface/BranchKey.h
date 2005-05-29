#ifndef EDM_BRANCHKEY_HH
#define EDM_BRANCHKEY_HH

/*----------------------------------------------------------------------
  
BranchKey: The key used to identify a Group in the EventPrincipal. The
name of the branch to which the related data product will be written
is determined entirely from the BranchKey.

$Id: BranchKey.h,v 1.6 2005/05/18 20:34:58 wmtan Exp $

----------------------------------------------------------------------*/
#include <iosfwd>
#include <string>
#include <utility>

#include "FWCore/CoreFramework/src/TypeID.h"

namespace edm
{
  struct BranchKey {
    BranchKey(TypeID id, const std::string& ml, const std::string& pn) :
      friendly_class_name(id.friendlyClassName()), 
      module_label(ml), 
      process_name(pn) 
    { }

    BranchKey(const std::string& cn, const std::string& ml, const std::string& pn) :
      friendly_class_name(cn), 
      module_label(ml), 
      process_name(pn) 
    { }

    std::string friendly_class_name;
    std::string module_label;
    std::string process_name; // ???
  };

  inline
  bool 
  operator<(const BranchKey& a, const BranchKey& b) {
      return 
	a.friendly_class_name < b.friendly_class_name ? true :
	a.friendly_class_name > b.friendly_class_name ? false :
	a.module_label < b.module_label ? true :
	a.module_label > b.module_label ? false :
	a.process_name < b.process_name ? true :
	false;
    }

  std::ostream&
  operator<<(std::ostream& os, const BranchKey& bk);
}
#endif
