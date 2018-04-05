#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"
#include <string>

template <>
std::string SiStripConfObject::get<std::string>( const std::string & name ) const
{
  std::string returnValue;
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

template <>
bool SiStripConfObject::put<std::vector<int> >( const std::string & name, const std::vector<int> & inputValue )
{
  std::stringstream ss;
  for(std::vector<int>::const_iterator elem=inputValue.begin();elem!=inputValue.end();++elem) {
    ss << *elem << " ";
  }
  if( parameters.insert(std::make_pair(name, ss.str())).second ) return true;
  return false;
}

template <>
bool SiStripConfObject::update<std::vector<int> >( const std::string & name, const std::vector<int> & inputValue )
{
  parMap::iterator it = parameters.find(name);
  if (it == parameters.end()) {
    std::cout << "WARNING in SiStripConfObject::update: parameter " << name << " not found, "
	      << "so cannot be updated to the vector of int of size'" << inputValue.size() << "'." << std::endl;
    return false;
  } else {
    std::stringstream ss;
    for(std::vector<int>::const_iterator elem=inputValue.begin();elem!=inputValue.end();++elem) {
      ss << *elem << " ";
    }
    it->second = ss.str();
    return true;
  }
}

template <>
std::vector<int> SiStripConfObject::get<std::vector<int> >( const std::string & name ) const
{
  std::vector<int> returnValue;
  parMap::const_iterator it = parameters.find(name);
  std::stringstream ss;
  if( it != parameters.end() ) {
    ss << it->second;
    int elem;
    while(ss >> elem) returnValue.push_back(elem);
  }
  else {
    std::cout << "WARNING: parameter " << name << " not found. Returning default value" << std::endl;
  }
  return returnValue;
}

template <>
bool SiStripConfObject::put<std::vector<std::string> >( const std::string & name, const std::vector<std::string> & inputValue )
{
  std::stringstream ss;
  for(std::vector<std::string>::const_iterator elem=inputValue.begin();elem!=inputValue.end();++elem) {
    ss << *elem << " ";
  }
  if( parameters.insert(std::make_pair(name, ss.str())).second ) return true;
  return false;
}

template <>
bool SiStripConfObject::update<std::vector<std::string> >( const std::string & name, const std::vector<std::string> & inputValue )
{
  parMap::iterator it = parameters.find(name);
  if (it == parameters.end()) {
    std::cout << "WARNING in SiStripConfObject::update: parameter " << name << " not found, "
	      << "so cannot be updated to the vector of std::string of size'" << inputValue.size() << "'." << std::endl;
    return false;
  } else {
    std::stringstream ss;
    for(std::vector<std::string>::const_iterator elem=inputValue.begin();elem!=inputValue.end();++elem) {
      ss << *elem << " ";
    }
    it->second = ss.str();
    return true;
  }
}

template <>
std::vector<std::string> SiStripConfObject::get<std::vector<std::string> >( const std::string & name ) const
{
  std::vector<std::string> returnValue;
  parMap::const_iterator it = parameters.find(name);
  std::stringstream ss;
  if( it != parameters.end() ) {
    ss << it->second;
    std::string elem;
    while(ss >> elem) returnValue.push_back(elem);
  }
  else {
    std::cout << "WARNING: parameter " << name << " not found. Returning default value" << std::endl;
  }
  return returnValue;
}


void SiStripConfObject::printSummary(std::stringstream & ss, const TrackerTopology* trackerTopo) const
{
  parMap::const_iterator it = parameters.begin();
  for( ; it != parameters.end(); ++it ) {
    ss << "parameter name = " << it->first << " value = " << it->second << std::endl;
  }
}

void SiStripConfObject::printDebug(std::stringstream & ss, const TrackerTopology* trackerTopo) const
{
  printSummary(ss, trackerTopo);
}
