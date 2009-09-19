#ifndef Selector_h
#define Selector_h

/**
  \class    Selector Selector.h "CommonTools/Utils/interface/Selector.h"
  \brief    Implements a string-indexed bit_vector

  class template that implements an interface to Selector utilities. This
  allows the user to access individual cuts based on a string index.
  The user can then turn individual cuts on and off at will. 

  \author Salvatore Rappoccio
  \version  $Id: Selector.h,v 1.20 2009/09/18 15:49:29 srappocc Exp $
*/


#include "CommonTools/Utils/interface/strbitset.h"
#include <fstream>
#include <functional>

/// Functor that operates on <T>
template<class T>
class Selector : public std::unary_function<T, bool>  {
  
 public:

  /// Constructor clears the bits
  Selector( ) { bits_.clear();}

  /// This is the registration of an individual cut string
  void push_back( std::string s) {
    bits_.push_back(s);
  }

  /// This provides the interface for base classes to select objects
  virtual bool operator()( T const & t) const  = 0;
  
  /// Set a given selection cut, on or off
  void set(std::string s, bool val = true) {
    bits_[s] = val;
  }

  /// Turn off a given selection cut. 
  void clear(std::string s) {
    bits_[s] = false;
  }

  bool operator[] ( std::string s ) const {
    return bits_[s];
  }

  /// Print the bitset
  void print(std::ostream & out) const { bits_.print(out);
  }

 protected:
  strbitset     bits_; //!< the bitset indexed by strings
};

#endif
