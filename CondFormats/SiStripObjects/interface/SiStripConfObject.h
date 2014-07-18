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
 * It stores a map<std::string, std::string> with all the parameters and their values. <br>
 * The put and get methods are provided to store and access the parameters. <br>
 * The put method retuns a bool which is true if the insertion was successuful. If the parameter
 * is already existing the insertion will not happen and the return value will be false. <br>
 * The get method is templated and works like the getParameter<type> of the framework. <br>
 * The isParameter method can be used to check whether a parameter exists. It will return a
 * bool with the result. <br>
 * The printSummary and printDebug method return both the full list of parameters. <br>
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


  /// Updating the value stored as 'name' with 'inputValue'. 
  /// False if parameter 'name' does not exist (and nothing is done then - use put(..) instead!),
  /// otherwise true.
  template <class valueType>
  bool update( const std::string & name, const valueType & inputValue )
  {
    parMap::iterator it = parameters.find(name);
    if (it == parameters.end()) {
      std::cout << "WARNING in SiStripConfObject::update: parameter " << name << " not found, "
		<< "so cannot be updated to '" << inputValue << "'." << std::endl;
      return false;
    } else {
      std::stringstream ss;
      ss << inputValue;
      it->second = ss.str();
      return true;
    }
  }

  template <class valueType>
  valueType get( const std::string & name ) const
  {
    valueType returnValue;
    parMap::const_iterator it = parameters.find(name);
    std::stringstream ss;
    if( it != parameters.end() ) {
      ss << it->second;
      ss >> returnValue;
    }
    else {
      std::cout << "WARNING: parameter " << name << " not found. Returning default value" << std::endl;
    }
    return returnValue;
  }


  bool isParameter( const std::string & name ) const
  {
    return( parameters.find(name) != parameters.end() );
  }

  /// Prints the full list of parameters
  void printSummary(std::stringstream & ss) const;
  /// Prints the full list of parameters
  void printDebug(std::stringstream & ss) const;

  typedef std::map<std::string, std::string> parMap;

  parMap parameters;
};

template <>
bool SiStripConfObject::put<std::vector<int> >( const std::string & name, const std::vector<int> & inputValue );
template <>
bool SiStripConfObject::update<std::vector<int> >( const std::string & name, const std::vector<int> & inputValue );
template <>
std::vector<int> SiStripConfObject::get<std::vector<int> >( const std::string & name ) const;
template <>
bool SiStripConfObject::put<std::vector<std::string> >( const std::string & name, const std::vector<std::string> & inputValue );
template <>
bool SiStripConfObject::update<std::vector<std::string> >( const std::string & name, const std::vector<std::string> & inputValue );
template <>
std::vector<std::string> SiStripConfObject::get<std::vector<std::string> >( const std::string & name ) const;


#endif
