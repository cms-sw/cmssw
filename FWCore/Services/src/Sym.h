#ifndef FWCore_Services_Sym_h
#define FWCore_Services_Sym_h

#include <iosfwd>
#include <string>

#include <dlfcn.h>

struct Sym {
  typedef void* address_type;

  Sym(Dl_info const& /*info*/, void* addr) :
    name_(),
    library_(),
    id_(),
    addr_(reinterpret_cast<address_type>(addr)) {
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
  address_type addr_;

  static int next_id_;

  bool
  operator<(address_type b) const
  { return addr_ < b; }

  bool
  operator<(const Sym& b) const
  { return addr_ < b.addr_; }
};

std::ostream&
operator<<(std::ostream& os, Sym const& s);

inline
bool
operator<(Sym::address_type a, const Sym& b)
{ return a < b.addr_; }

#endif
