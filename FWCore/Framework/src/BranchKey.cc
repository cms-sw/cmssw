/*----------------------------------------------------------------------
  
$Id: BranchKey.cc,v 1.3 2005/05/18 20:34:58 wmtan Exp $

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
       << bk.process_name << ')';
    return os;
  }

  
}
