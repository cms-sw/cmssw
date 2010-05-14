#ifndef SiStripConfObject_h
#define SiStripConfObject_h

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <iterator>
#include <sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

/**
 * Author M. De Mattia - 16/11/2009
 *
 * Simple class used to store configuration values. <br>
 * It stores a vector<string> and a vector<int> containing the name and value of the parameter. <br>
 * The put and get methods are provied to store and access the parameters. <br>
 * The printSummary and printDebug method return both the full list of parameters. <br>
 * The vectors with names and parameters are public. <br>
 * WARNING: the get method assumes that the elements in the two vectors correspond (vector<string>[i] <-> vector<int>[i]).
 * This is the case if the values are input with the put method.
 */

class SiStripConfObject
{
 public:
  SiStripConfObject() {}

  template <class valueType>
  bool put( const std::string & name, const valueType & inputValue )
  {
    std::stringstream ss;
    ss << inputValue;
    if( parameters.insert(std::make_pair(name, ss.str())).second ) return true;
    return false;
  }

  int getInt( const std::string & name );
  double getDouble( const std::string & name );
  std::string getString( const std::string & name );

  template <class valueType>
  void get( const std::string & name, valueType * value )
  {
    parMap::const_iterator it = parameters.find(name);
    std::stringstream ss;
    if( it != parameters.end() ) {
      ss << it->second;
      ss >> (*value);
    }
    else {
      std::cout << "WARNING: parameter " << name << " not found. Returning default value" << std::endl;
    }
  }

  /// Prints the full list of parameters
  void printSummary(std::stringstream & ss) const;
  /// Prints the full list of parameters
  void printDebug(std::stringstream & ss) const;

  typedef std::map<std::string, std::string> parMap;

  parMap parameters;
};

#endif
