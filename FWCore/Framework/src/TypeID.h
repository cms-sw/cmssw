#ifndef EDM_TYPEID_H
#define EDM_TYPEID_H

/*----------------------------------------------------------------------
  
TypeID: A unique identifier for a C++ type.

The identifier is unique within an entire program, but can not be
persisted across invocations of the program.

$Id: TypeID.h,v 1.6 2005/07/11 21:55:14 wmtan Exp $

----------------------------------------------------------------------*/
#include <iosfwd>
#include <typeinfo>
#include <string>
#include "FWCore/FWUtilities/interface/EDMException.h"

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

    // Print out the name of the type, using the reflection class name.
    void print(std::ostream& os) const;

    std::string reflectionClassName() const;

    std::string userClassName() const;

    std::string friendlyClassName() const;

  private:
    const std::type_info& t_;

    TypeID& operator=(const TypeID&); // not implemented
   
    static bool stripTemplate(std::string& name);

    static bool stripNamespace(std::string& name);

  };

  inline bool operator>(const TypeID& a, const TypeID& b)
  { return b<a; }
  
  inline bool operator!=(const TypeID& a, const TypeID& b)
  { return !(a==b); }

   std::ostream& operator<<(std::ostream& os, const TypeID& id);
}
#endif
