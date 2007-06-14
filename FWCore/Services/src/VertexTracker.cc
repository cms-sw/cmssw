#include "VertexTracker.h"

#include <ostream>

unsigned int VertexTracker::next_id_ = 0;

std::ostream& 
operator<<(std::ostream& ost, const VertexTracker& vt)
{
//   static int empty_count = 1;
//   std::string name = vt.name_;

  // this is a bad place for this code
//   if (name.empty())
//     {
//       std::ostringstream ostr;
//       ostr << "EmptyName-" << empty_count;
//       ++empty_count;
//       name = ostr.str();
//     }

  ost << vt.id_ << '\t'
      << (void*)vt.addr_ << '\t'
      << vt.total_as_leaf_ << '\t'
      << vt.total_seen_ << '\t'
      << vt.in_path_ << '\t'
      << vt.percent_leaf_ << '\t'
      << vt.percent_path_ << '\t'
      << '"' << vt.library_ << "\"\t"
      << '"' << vt.name_ << '"';

  return ost;
}
