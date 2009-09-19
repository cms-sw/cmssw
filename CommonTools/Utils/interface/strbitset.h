#ifndef strbitset_h
#define strbitset_h

/**
  \class    strbitset strbitset.h "CommonTools/Utils/interface/strbitset.h"
  \brief    Implements a string-indexed bit_vector

   The strbitset implements a string-indexed bit vector that will allow users
   to access the underlying bits by a string name instead of via an index.

  \author Salvatore Rappoccio
  \version  $Id: strbitset.h,v 1.20 2009/09/18 15:49:29 srappocc Exp $
*/


#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <fstream>

class strbitset {
 public:

  // Add typedefs
  typedef unsigned int                    size_t;
  typedef std::map<std::string, size_t>   str_index_map;
  typedef str_index_map::const_iterator   map_const_iterator;
  typedef str_index_map::iterator         map_iterator;
  typedef std::vector<bool>               bit_vector;
  typedef bit_vector::const_iterator      const_iterator;
  typedef bit_vector::iterator            iterator;
  typedef bit_vector::value_type          value_type;
  typedef bit_vector::reference           reference;
  typedef bit_vector::const_reference     const_reference;

  //! constructor: just clears the bitset and map
  strbitset() {
    clear();
  }

  //! clear the bitset and map
  void clear() {
    map_.clear();
    bits_.clear();
  }
  
  /// adds an item that is indexed by the string. this
  /// can then be sorted, cut, whatever, and the
  /// index mapping is kept
  void push_back( std::string s ) {
    map_[s] = bits_.size();
    bits_.resize( bits_.size() + 1 );
    *(bits_.rbegin()) = false;
  }


  //! access method const
  bit_vector::const_reference operator[] ( std::string s) const {
    size_t index = this->index(s);
    return bits_.operator[](index);
  }

  //! access method non-const
  bit_vector::reference operator[] ( std::string s) {
    size_t index = this->index(s);
    return bits_.operator[](index);
  }


  //! print method
  void print(std::ostream & out) const {
    for( map_const_iterator mbegin = map_.begin(),
	   mend = map_.end(),
	   mit = mbegin;
	 mit != mend; ++mit ) {
      char buff[100];
      sprintf(buff, "%10s = %6d", mit->first.c_str(), bits_.at(mit->second));
      out << buff << std::endl;
    }
  }

 private:

  /// workhorse: this gets the index of "bits" that is pointed to by
  /// the string "s"
  size_t  index(std::string s) const {
    map_const_iterator f = map_.find(s);
    if ( f == map_.end() ) {
      std::cout << "Cannot find " << s << ", returning size()" << std::endl;
      return map_.size();
    } else {
      return f->second;
    }
  }

  
  str_index_map     map_;   //!< map that holds the string-->index map 
  bit_vector        bits_;  //!< the actual bits, indexed by the index in "map_"
};

#endif
