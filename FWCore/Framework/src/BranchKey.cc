/*----------------------------------------------------------------------
  
$Id: BranchKey.cc,v 1.3 2005/07/14 22:50:53 wmtan Exp $

----------------------------------------------------------------------*/
#include <ostream>

#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/Framework/interface/ProductDescription.h"


namespace edm
{
  BranchKey::BranchKey(ProductDescription const& desc) :
    friendly_class_name(desc.friendly_product_type_name),
    module_label(desc.module.module_label),
    product_instance_name(desc.product_instance_name),
    process_name(desc.module.process_name) {}

  std::ostream&
  operator<<(std::ostream& os, const BranchKey& bk) {
    os << "BranchKey("
       << bk.friendly_class_name << ", "
       << bk.module_label << ", "
       << bk.product_instance_name << ", "
       << bk.process_name << ')';
    return os;
  }

  
}
