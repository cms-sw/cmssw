#ifndef PhysicsTools_SelectorUtils_Selector_h
#define PhysicsTools_SelectorUtils_Selector_h

/**
  \class    Selector Selector.h "CommonTools/Utils/interface/Selector.h"
  \brief    Implements a string-indexed bit_vector

  class template that implements an interface to Selector utilities. This
  allows the user to access individual cuts based on a string index.
  The user can then turn individual cuts on and off at will.

  \author Salvatore Rappoccio
*/


#include "PhysicsTools/SelectorUtils/interface/strbitset.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Common/interface/EventBase.h"
#include <fstream>
#include <functional>

/// Functor that operates on <T>
template<class T>
class Selector : public std::binary_function<T, pat::strbitset, bool>  {

 public:
  typedef T                                            data_type;
  typedef std::binary_function<T,pat::strbitset,bool>  base_type;
  typedef pat::strbitset::index_type                   index_type;
  typedef std::pair<index_type, size_t>                cut_flow_item;
  typedef std::vector<cut_flow_item>                   cut_flow_map;
  typedef std::map<index_type, int>                    int_map;
  typedef std::map<index_type, double>                 double_map;

  /// Constructor clears the bits
  Selector( ) {
    bits_.clear();
    intCuts_.clear();
    doubleCuts_.clear();
    cutFlow_.clear();
    retInternal_ = getBitTemplate();
  }
  virtual ~Selector() {}

  /// This is the registration of an individual cut string
  virtual void push_back( std::string const & s) {
    bits_.push_back(s);
    index_type i(&bits_,s);
    // don't need to check to see if the key is already there,
    // bits_ does that.
    cutFlow_.push_back( cut_flow_item(i, 0) );
  }


  /// This is the registration of an individual cut string, with an int cut value
  virtual void push_back( std::string const & s, int cut) {
    bits_.push_back(s);
    index_type i(&bits_,s);
    intCuts_[i] = cut;
    // don't need to check to see if the key is already there,
    // bits_ does that.
    cutFlow_.push_back( cut_flow_item(i,0) );
  }

  /// This is the registration of an individual cut string, with a double cut value
  virtual void push_back( std::string const & s, double cut) {
    bits_.push_back(s);
    index_type i(&bits_,s);
    doubleCuts_[i] = cut;
    // don't need to check to see if the key is already there,
    // bits_ does that.
    cutFlow_.push_back( cut_flow_item(i,0) );
  }

  /// This provides the interface for base classes to select objects
  virtual bool operator()( T const & t, pat::strbitset & ret ) = 0;

  /// This provides an alternative signature without the second ret
  virtual bool operator()( T const & t )
  {
    retInternal_.set(false);
    operator()(t, retInternal_);
    setIgnored(retInternal_);
    return (bool)retInternal_;
  }


  /// This provides an alternative signature that includes extra information
  virtual bool operator()( T const & t, edm::EventBase const & e, pat::strbitset & ret)
  {
    return operator()(t, ret);
  }

  /// This provides an alternative signature that includes extra information
  virtual bool operator()( T const & t, edm::EventBase const & e)
  {
    retInternal_.set(false);
    operator()(t, e, retInternal_);
    setIgnored(retInternal_);
    return (bool)retInternal_;
  }


  /// Set a given selection cut, on or off
  void set(std::string const & s, bool val = true) {
    set( index_type(&bits_,s), val);
  }
  void set(index_type const & i, bool val = true) {
    bits_[i] = val;
  }

  /// Set a given selection cut, on or off, and reset int cut value
  void set(std::string const & s, int cut, bool val = true) {
    set( index_type(&bits_,s), cut);
  }
  void set(index_type const & i, int cut, bool val = true) {
    bits_[i] = val;
    intCuts_[i] = cut;
  }

  /// Set a given selection cut, on or off, and reset int cut value
  void set(std::string const & s, double cut, bool val = true) {
    set( index_type(&bits_,s), cut);
  }
  void set(index_type const & i, double cut, bool val = true) {
    bits_[i] = val;
    doubleCuts_[i] = cut;
  }

  /// Turn off a given selection cut.
  void clear(std::string const & s) {
    clear(index_type(&bits_,s));
  }

  void clear(index_type const & i) {
    bits_[i] = false;
  }

  /// Access the selector cut at index "s".
  /// "true" means to consider the cut.
  /// "false" means to ignore the cut.
  bool operator[] ( std::string const & s ) const {
    return bits_[s];
  }

  bool operator[] ( index_type const & i ) const {
    return bits_[i];
  }

  /// consider the cut at index "s"
  bool considerCut( std::string const & s ) const {
    return bits_[s] == true;
  }
  bool considerCut( index_type const & i  ) const {
    return bits_[i] == true;
  }

  /// ignore the cut at index "s"
  bool ignoreCut( std::string const & s ) const {
    return bits_[s] == false;
  }
  bool ignoreCut( index_type const & i ) const {
    return bits_[i] == false;
  }

  /// set the bits to ignore from a vector
  void setIgnoredCuts( std::vector<std::string> const & bitsToIgnore ) {
    for ( std::vector<std::string>::const_iterator ignoreBegin = bitsToIgnore.begin(),
	    ignoreEnd = bitsToIgnore.end(), ibit = ignoreBegin;
	  ibit != ignoreEnd; ++ibit ) {
      set(*ibit, false );
    }
  }

  /// Passing cuts
  void passCut( pat::strbitset & ret, std::string const & s ) {
    passCut( ret, index_type(&bits_,s));
  }

  void passCut( pat::strbitset & ret, index_type const & i ) {
    ret[i] = true;
    cut_flow_map::iterator found = cutFlow_.end();
    for ( cut_flow_map::iterator cutsBegin = cutFlow_.begin(),
	    cutsEnd = cutFlow_.end(), icut = cutsBegin;
	  icut != cutsEnd && found == cutsEnd; ++icut ) {
      if ( icut->first == i ) {
	found = icut;
      }
    }    
    ++(found->second);
  }

  /// Access the int cut values at index "s"
  int cut( index_type const & i, int val ) const {
    return intCuts_.find( i )->second;
  };
  /// Access the double cut values at index "s"
  double cut( index_type const & i, double val ) const {
    return doubleCuts_.find( i )->second;
  };

  /// Access the int cut values at index "s"
  int cut( std::string s, int val ) const {
    return cut( index_type(&bits_,s), val);
  };
  /// Access the double cut values at index "s"
  double cut( std::string s, double val ) const {
    return cut( index_type(&bits_,s), val);
  };

  /// Get an empty bitset with the proper names
  pat::strbitset getBitTemplate() const {
    pat::strbitset ret = bits_;
    ret.set(false);
    for ( cut_flow_map::const_iterator cutsBegin = cutFlow_.begin(),
	    cutsEnd = cutFlow_.end(), icut = cutsBegin;
	  icut != cutsEnd; ++icut ) {
      if ( ignoreCut(icut->first) ) ret[icut->first] = true;
    }
    return ret;
  }

  /// set ignored bits
  void setIgnored( pat::strbitset & ret ) {
    for ( cut_flow_map::const_iterator cutsBegin = cutFlow_.begin(),
	    cutsEnd = cutFlow_.end(), icut = cutsBegin;
	  icut != cutsEnd; ++icut ) {
      if ( ignoreCut(icut->first) ) ret[icut->first] = true;
    }
  }

  /// Print the cut flow
  void print(std::ostream & out) const {
    for ( cut_flow_map::const_iterator cutsBegin = cutFlow_.begin(),
	    cutsEnd = cutFlow_.end(), icut = cutsBegin;
	  icut != cutsEnd; ++icut ) {
      char buff[1000];
      if ( considerCut( icut->first ) ) {
	sprintf(buff, "%6lu : %20s %10lu",
		static_cast<unsigned long>(icut - cutsBegin),
		icut->first.str().c_str(),
		static_cast<unsigned long>(icut->second) );
      } else {
	sprintf(buff, "%6lu : %20s %10s",
		static_cast<unsigned long>(icut - cutsBegin),
		icut->first.str().c_str(),
		"off" );
      }
      out << buff << std::endl;
    }
  }

  /// Print the cuts being considered
  void printActiveCuts(std::ostream & out) const {
    bool already_printed_one = false;
    for ( cut_flow_map::const_iterator cutsBegin = cutFlow_.begin(),
	    cutsEnd = cutFlow_.end(), icut = cutsBegin;
	  icut != cutsEnd; ++icut ) {
      if ( considerCut( icut->first ) ) {
	if( already_printed_one ) out << ", ";
	out << icut->first;
	already_printed_one = true;
      }
    }
    out << std::endl;
  }

  /// Return the number of passing cases
  double getPasses( std::string const & s ) const {
    return getPasses( index_type(&bits_,s) );
  }
  double getPasses( index_type const &i ) const {
    cut_flow_map::const_iterator found = cutFlow_.end();
    for ( cut_flow_map::const_iterator cutsBegin = cutFlow_.begin(),
            cutsEnd = cutFlow_.end(), icut = cutsBegin;
          icut != cutsEnd && found == cutsEnd; ++icut ) {
      if ( icut->first == i ) {
	found = icut;
      }
    }
    return found->second;
  }


 protected:
  pat::strbitset bits_;        //!< the bitset indexed by strings
  pat::strbitset retInternal_; //!< internal ret if users don't care about return bits
  int_map        intCuts_;     //!< the int-value cut map
  double_map     doubleCuts_;  //!< the double-value cut map
  cut_flow_map   cutFlow_;     //!< map of cut flows in "human" order
};

#endif
