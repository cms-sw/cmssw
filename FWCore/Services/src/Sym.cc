#include "Sym.h"

#include <ostream>


std::ostream& 
operator<<(std::ostream& ost,const Sym& s)
{
  ost << s.id_ << " " << s.addr_ << " " << s.name_;
  return ost;
}

int Sym::next_id_ = 1000000;
