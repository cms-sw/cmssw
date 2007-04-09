#ifndef FWCORE_SERVICES_SYM_H
#define FWCORE_SERVICES_SYM_H

#include <iosfwd>
#include <string>

struct Sym
{
  Sym() :
    id_(),
    addr_() { }

  explicit Sym(int id) :
    id_(id),
    addr_()
  { }
  
  int id_;
  unsigned int addr_;
  std::string name_;
  int size_;

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
