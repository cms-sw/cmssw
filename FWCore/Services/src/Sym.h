#ifndef FWCORE_SERVICES_SYM_H
#define FWCORE_SERVICES_SYM_H

#include <iosfwd>
#include <string>

#include <dlfcn.h>

struct Sym
{
  Sym(Dl_info const& info, void* addr) :
    name_(),
    library_(),
    id_(),
    addr_(reinterpret_cast<unsigned int>(addr))
  {
    
  }



  Sym() :
    name_(),
    library_(),
    id_(),
    addr_()
  {  }

  explicit Sym(int id) :
    name_(),
    library_(),
    id_(id),
    addr_()
  { }
  
  std::string  name_;
  std::string  library_;
  int          id_;
  unsigned int addr_;

  static int next_id_;

  bool 
  operator<(unsigned int b) const
  { return addr_ < b; }

  bool 
  operator<(const Sym& b) const
  { return addr_ < b.addr_; }
};

std::ostream&
operator<< (std::ostream& os, Sym const& s);

inline 
bool 
operator<(unsigned int a, const Sym& b)
{ return a < b.addr_; }


#endif 
