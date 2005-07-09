/*----------------------------------------------------------------------
  
$Id: BranchKey.cc,v 1.1 2005/05/29 02:29:53 wmtan Exp $

----------------------------------------------------------------------*/
#include <ostream>

#include "FWCore/CoreFramework/interface/BranchKey.h"


namespace edm
{

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
