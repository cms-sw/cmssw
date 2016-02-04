#ifndef DataFormats_Provenance_Transient_h
#define DataFormats_Provenance_Transient_h
// -*- C++ -*-
//
// Package:     Provenance
// Class  :     Transient
// 
/**\class Transient Transient.h DataFormats/Provenance/interface/Transient.h

 Description: ROOT safe bool

 Usage:
    We define a template for transients  in order to guarantee that value_
    is always reset when ever a new instance of this class is read from a file.

*/
//
// Original Author:  Bill Tanenbaum
//         Created:  Sat Aug 18 17:30:08 EDT 2007
//

// system include files

// user include files

// forward declarations
namespace edm {

template <typename T>
class Transient {
public:
  typedef T value_type;
  Transient() : value_(T()) {}
  explicit Transient(T const& value) : value_(value) {}
  operator T() const { return value_; }
  Transient & operator=(T const& rh) { value_ = rh; return *this; }
  T const& get() const { return value_;}
  T & get() { return value_;}
private:
  T value_;
};
}
#endif
