/*----------------------------------------------------------------------
  
$Id: BranchKey.cc,v 1.2 2005/07/09 02:08:15 wmtan Exp $

----------------------------------------------------------------------*/
#include <ostream>

#include "FWCore/Framework/interface/BranchKey.h"


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
