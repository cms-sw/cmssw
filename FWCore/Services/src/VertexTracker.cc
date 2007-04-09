#include "VertexTracker.h"

#include <ostream>
#include <sstream>

unsigned int VertexTracker::next_id_ = 0;

std::ostream& 
operator<<(std::ostream& ost, const VertexTracker& a)
{
  static int empty_count = 1;
  std::string name = a.name_;

  // this is a bad place for this code
  if (name.empty())
    {
      std::ostringstream ostr;
      ostr << "EmptyName-" << empty_count;
      ++empty_count;
      name = ostr.str();
    }

  ost << a.id_ << '\t'
      << (void*)a.addr_ << '\t'
      << a.total_as_leaf_ << '\t'
      << a.total_seen_ << '\t'
      << a.in_path_ << '\t'
      << a.percent_leaf_ << '\t'
      << a.percent_path_ << '\t'
      << '"' << name << '"';

  return ost;
}
