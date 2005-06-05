#ifndef EDM_TYPEID_H
#define EDM_TYPEID_H

/*----------------------------------------------------------------------
  
TypeID: A unique identifier for a C++ type.

The identifier is unique within an entire program, but can not be
persisted across invocations of the program.

$Id: TypeID.h,v 1.1 2005/05/29 02:29:54 wmtan Exp $

----------------------------------------------------------------------*/
#include <iosfwd>
#include <typeinfo>
#include <string>
#include "Reflection/Class.h"
#include <stdexcept>

namespace edm {

  class TypeID
  {
  public:
    struct Def { };

    TypeID() :
      t_(typeid(Def)) 
    { }

    TypeID(const TypeID& other) :
      t_(other.t_)
    { }
    
    explicit TypeID(const std::type_info& t) :
      t_(t)
    { }

    // Copy assignment disallowed; see below.

    template <typename T>
    explicit TypeID(const T& t) :
      t_(typeid(t))
    { }

    // Returned C-style string owned by system; do not delete[] it.
    // This is the (horrible, mangled, platform-dependent) name of
    // the type.
    const char* name() const { return t_.name(); }

    bool operator<(const TypeID& b) const { return t_.before(b.t_); }
    bool operator==(const TypeID& b) const { return t_ == b.t_; }

    // Print out the (horrible, mangled, platform-dependent) name of
    // the type.
    void print(std::ostream& os) const;

    std::string reflectionClassName() const {
      seal::reflect::Class const * c = seal::reflect::Class::forTypeinfo(t_);
      if ( c == 0 ) {
	std::string error( "No SEAL Reflection entry for class: " );
	error +=  t_.name();
	throw std::runtime_error( error.c_str() );
      }
      return c->fullName();
    }

    std::string friendlyClassName() const {
      std::string name = reflectionClassName();
      if (name.find("edm::EDCollection") == 0) {
	std::string::size_type idx = name.find('<');
	std::string::size_type idx2 = name.rfind('>');
	assert (idx != std::string::npos);
	assert (idx2 != std::string::npos);
	++idx;
	name = name.substr(idx, idx2 - idx);
	name += "Collection";
      }
      return name;
    }

  private:
    const std::type_info& t_;

    TypeID& operator=(const TypeID&); // not implemented
  };

  inline bool operator>(const TypeID& a, const TypeID& b)
  { return b<a; }
  
  inline bool operator!=(const TypeID& a, const TypeID& b)
  { return !(a==b); }

   std::ostream& operator<<(std::ostream& os, const TypeID& id);
}
#endif
