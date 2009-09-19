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
  typedef std::map<size_t, int>      int_map;
  typedef std::map<size_t, double>   double_map;

  /// Constructor clears the bits
  Selector( ) { bits_.clear(); intCuts_.clear(); doubleCuts_.clear();}

  virtual ~Selector() {}

  /// This is the registration of an individual cut string
  void push_back( std::string s) {
    bits_.push_back(s);
  }

  /// This is the registration of an individual cut string, with an int cut value
  void push_back( std::string s, int cut) {
    bits_.push_back(s);
    intCuts_[bits_[s]] = cut;
  }

  /// This is the registration of an individual cut string, with a double cut value
  void push_back( std::string s, double cut) {
    bits_.push_back(s);
    doubleCuts_[bits_[s]] = cut;
  }

  /// This provides the interface for base classes to select objects
  virtual bool operator()( T const & t) const  = 0;
  
  /// Set a given selection cut, on or off
  void set(std::string s, bool val = true) {
    bits_[s] = val;
  }

  /// Set a given selection cut, on or off, and reset int cut value
  void set(std::string s, int cut, bool val = true) {
    bits_[s] = val;
    intCuts_[bits_[s]] = cut;
  }

  /// Set a given selection cut, on or off, and reset int cut value
  void set(std::string s, double cut, bool val = true) {
    bits_[s] = val;
    doubleCuts_[bits_[s]] = cut;
  }

  /// Turn off a given selection cut. 
  void clear(std::string s) {
    bits_[s] = false;
  }

  /// Access the selector cut at index "s"
  bool operator[] ( std::string s ) const {
    return bits_[s];
  }

  /// Access the int cut values at index "s"
  int cut( std::string s, int val ) const {
    return intCuts_.find( bits_[s] )->second;
  };

  /// Access the double cut values at index "s"
  double cut( std::string s, double val ) const {
    return doubleCuts_.find( bits_[s] )->second;
  };

  /// Print the bitset
  void print(std::ostream & out) const { bits_.print(out);
  }

 protected:
  strbitset     bits_;        //!< the bitset indexed by strings
  int_map       intCuts_;     //!< the int-value cut map
  double_map    doubleCuts_;  //!< the double-value cut map
};

#endif
