#ifndef EDM_BRANCHKEY_HH
#define EDM_BRANCHKEY_HH

/*----------------------------------------------------------------------
  
BranchKey: The key used to identify a Group in the EventPrincipal. The
name of the branch to which the related data product will be written
is determined entirely from the BranchKey.

$Id: BranchKey.h,v 1.5 2005/07/25 04:00:52 wmtan Exp $

----------------------------------------------------------------------*/
#include <iosfwd>
#include <string>
#include <utility>

#include "FWCore/Framework/src/TypeID.h"

namespace edm
{
  struct BranchKey {
    BranchKey() : friendly_class_name(), module_label(), product_instance_name(), process_name()
    {}

    BranchKey(TypeID id, std::string const& ml, std::string const& pin, std::string const& pn) :
      friendly_class_name(id.friendlyClassName()), 
      module_label(ml), 
      product_instance_name(pin), 
      process_name(pn) 
    {}

    BranchKey(std::string const& cn, std::string const& ml,
        std::string const& pin, std::string const& pn) :
      friendly_class_name(cn), 
      module_label(ml), 
      product_instance_name(pin), 
      process_name(pn) 
    {}

    std::string friendly_class_name;
    std::string module_label;
    std::string product_instance_name;
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
	a.product_instance_name < b.product_instance_name ? true :
	a.product_instance_name > b.product_instance_name ? false :
	a.process_name < b.process_name ? true :
	false;
    }

  std::ostream&
  operator<<(std::ostream& os, const BranchKey& bk);
}
#endif
