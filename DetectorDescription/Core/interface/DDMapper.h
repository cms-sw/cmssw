#ifndef DD_DDMapper
#define DD_DDMapper

#include <map>
#include <string>
#include <vector>

#include <iostream>

#include "DetectorDescription/Core/interface/DDsvalues.h"

template<class KeyType, class ValueType>
class DDMapper
{
public:  
  //! usefull typedef
  typedef std::pair<KeyType, ValueType> Pair;

  //! usefull typedef
  typedef std::vector<Pair> Vector; 

  //! insert a new key-value-pair
  /** any existing entry will be overridden */
  void insert(const KeyType &, const ValueType &);
  
  //! removes a key-value pair
  /** non-existing keys are simply ignored */
  //void remove(const KeyType &);

  //! removes a key-value pair
  /** non-existing values are simply ignored */
  //void remove(const ValueType &);
  
  //! fetch a value given a key
  /** returns true, if sucessfull - value is assigned to result; else false */
  bool value(const KeyType & key, ValueType & result);
  
  //! fetch a key given a value
  //bool keys(const ValueType & value, std::vector<KeyType> & result);
    
  //! the number of specific parameters which are named 'name'
  unsigned int noSpecifics(const KeyType & key, const std::string & name) const;
  
  //! returns the number specific parameters named 'name' and the corrsponding double 
  /** of the DDLogicalPart which corresponds to the given KeyType key. The returned value
     is assigned to the 'value'-reference 
     - if no parameters exist, 0 is returned and value is left unchanged
     - if more than one parameter with the given name exists, the first is returned by default,
       alternatively 'pos' can be used to address another value (note: pos=0 -> first value) */     
  unsigned int toDouble(const std::string & name, const KeyType & key, double & value, unsigned int pos=0) const;
  
  //! same as toDouble but for std::string-valued values of named parameters
  unsigned int toString(const std::string & name, const KeyType & key, std::string & value, unsigned int pos=0) const;

  unsigned int toDouble(const std::string & name, const ValueType & key, double & value, unsigned int pos=0) const;
  
  //! same as toDouble but for std::string-valued values of named parameters
  unsigned int toString(const std::string & name, const ValueType & key, std::string & value, unsigned int pos=0) const;
    
  //! get all std::mapped instances which have a specific 'name' with value 'value'
  Vector all(const std::string & name, const std::string & value) const;
  
  //! get all std::mapped instances which have a specific 'name' with value 'value'
  Vector all(const std::string & name, const double & value) const;

  //! get all std::mapped instances which have a specific 'name'
  Vector all(const std::string & name) const;


private:
  std::map<KeyType, ValueType> keyToValue_;
  std::multimap<ValueType, KeyType> valueToKey_;  
};


template<class K, class V>
void DDMapper<K,V>::insert(const K & key, const V & value)
{
   keyToValue_[key] = value;
   valueToKey_.insert(std::make_pair(value,key));
   //   valueToKey_[value] = key;
}


template<class K, class V>
bool DDMapper<K,V>::value(const K & key, V & value)
{
  bool result = false;
  typename std::map<K,V>::const_iterator it = keyToValue_.find(key);
  if (it != keyToValue_.end()) {
    value = it->second;
    result = true;
  }
  return result;
}

template<class K, class V>
unsigned int DDMapper<K,V>::noSpecifics(const K & key, const std::string & name) const
{
  typedef std::vector<const DDsvalues_type *> sv_type;
  unsigned int result = 0;
  typename std::map<K,V>::const_iterator it = keyToValue_.find(key);
  if (it != keyToValue_.end()) {
    sv_type sv = it->second.specifics();
    sv_type::const_iterator it = sv.begin();
    DDValue v(name);
    for (; it != sv.end(); ++it) {
      if (DDfetch(*it, v)) {
        result += v.size();
      }
    }
  }
  return result;
}


// to be fast, we will only return the first specific found matching the given name
template<class K, class V>
unsigned int DDMapper<K,V>::toDouble(const std::string & name, const K & key, double & value, unsigned int pos) const
{
  typedef std::vector<const DDsvalues_type *> sv_type;
  unsigned int result=0;
  typename std::map<K,V>::const_iterator it = keyToValue_.find(key);
  if (it != keyToValue_.end()) {
    sv_type sv = it->second.specifics();
    sv_type::const_iterator svIt = sv.begin();
    sv_type::const_iterator svEd = sv.end();
    DDValue v(name);
    for (; svIt != svEd; ++svIt) {
      if (DDfetch(*svIt,v)) {
        result = v.size();
        value = v.doubles()[pos];
	 break;
      }	
    } 
  }  
  return result;
}


template<class K, class V>
unsigned int DDMapper<K,V>::toDouble(const std::string & name, const V & val, double & value, unsigned int pos) const
{
  typedef std::vector<const DDsvalues_type *> sv_type;
    unsigned int result=0;
    sv_type sv = val.specifics();
    sv_type::const_iterator svIt = sv.begin();
    sv_type::const_iterator svEd = sv.end();
    DDValue v(name);
    for (; svIt != svEd; ++svIt) {
      if (DDfetch(*svIt,v)) {
        result = v.size();
        value = v.doubles()[pos];
	 break;
      }	
    }    
    return result;
}


template<class K, class V>
unsigned int DDMapper<K,V>::toString(const std::string & name, const V & val, std::string & value, unsigned int pos) const
{
  typedef std::vector<const DDsvalues_type *> sv_type;
    unsigned int result=0;
    sv_type sv = val.specifics();
    sv_type::const_iterator svIt = sv.begin();
    sv_type::const_iterator svEd = sv.end();
    DDValue v(name);
    for (; svIt != svEd; ++svIt) {
      if (DDfetch(*svIt,v)) {
        result = v.size();
        value = v.strings()[pos];
	 break;
      }	
    }    
    return result;
}

// to be fast, we will only return the first specific found matcing the given name
template<class K, class V>
unsigned int DDMapper<K,V>::toString(const std::string & name, const K & key, std::string & value, unsigned int pos) const
{
  typedef std::vector<const DDsvalues_type *> sv_type;
  unsigned int result=0;
  typename std::map<K,V>::const_iterator it = keyToValue_.find(key);
  if (it != keyToValue_.end()) {
    sv_type sv = it->second.specifics();
    sv_type::const_iterator svIt = sv.begin();
    sv_type::const_iterator svEd = sv.end();
    DDValue v(name);
    //std::cout << "DDValue=" << name << std::endl;
    for (; svIt != svEd; ++svIt) {
      //std::cout << "looping..." << **svIt << std::endl;
      if (DDfetch(*svIt,v)) {
        result = v.size();
	 //std::cout << "found!" << std::endl;
        value = v.strings()[pos];
	 break;
      }	
    }
  }  
  return result;
}


template<class K, class V>
std::vector<std::pair<K,V> > DDMapper<K,V>::all(const std::string & name, const std::string & value) const
{
  std::vector<std::pair<K,V> > result;
  typedef std::vector<const DDsvalues_type *> sv_type;
  typename std::map<V,K>::const_iterator it = valueToKey_.begin();
  typename std::map<V,K>::const_iterator ed = valueToKey_.end();
  
  // loop over all registered ValueTypes
  for (; it != ed; ++it) {
     sv_type sv = it->first.specifics();
     //std::cout << "now at: " << it->first.name() << std::endl;
     sv_type::const_iterator svIt = sv.begin();
     sv_type::const_iterator svEd = sv.end();
     DDValue v(name);
     for (; svIt != svEd; ++svIt) {
       if (DDfetch(*svIt,v)) {
         //std::cout << "found: ";
         const std::vector<std::string> & s = v.strings();
  	  if (s.size()) {
	    //std::cout << s[0];
	    if (s[0]==value) {
	      result.push_back(std::make_pair(it->second,it->first));
	      break;
	    }
	  }	  
	 //std::cout << std::endl; 
       }
     }  
  }
  return result;
}


template<class K, class V>
std::vector<std::pair<K,V> > DDMapper<K,V>::all(const std::string & name, const double & value) const
{
  std::vector<std::pair<K,V> > result;
  typedef std::vector<const DDsvalues_type *> sv_type;
  typename std::map<V,K>::const_iterator it = valueToKey_.begin();
  typename std::map<V,K>::const_iterator ed = valueToKey_.end();
  
  // loop over all registered ValueTypes
  for (; it != ed; ++it) {
     sv_type sv = it->first.specifics();
     //std::cout << "now at: " << it->first.name() << std::endl;
     sv_type::const_iterator svIt = sv.begin();
     sv_type::const_iterator svEd = sv.end();
     DDValue v(name);
     for (; svIt != svEd; ++svIt) {
       if (DDfetch(*svIt,v)) {
         //std::cout << "found: ";
         const std::vector<double> & s = v.doubles();
  	  if (s.size()) {
	    //std::cout << s[0];
	    if (s[0]==value) {
	      result.push_back(std::make_pair(it->second,it->first));
	      break;
	    }
	  }	  
	 //std::cout << std::endl; 
       }
     }  
  }
  return result;
}


template<class K, class V>
std::vector<std::pair<K,V> > DDMapper<K,V>::all(const std::string & name) const
{
  std::vector<std::pair<K,V> > result;
  typedef std::vector<const DDsvalues_type *> sv_type;
  typename std::map<V,K>::const_iterator it = valueToKey_.begin();
  typename std::map<V,K>::const_iterator ed = valueToKey_.end();
  
  // loop over all registered ValueTypes
  for (; it != ed; ++it) {
     sv_type sv = it->first.specifics();
     //std::cout << "now at: " << it->first.name() << std::endl;
     sv_type::const_iterator svIt = sv.begin();
     sv_type::const_iterator svEd = sv.end();
     DDValue v(name);
     for (; svIt != svEd; ++svIt) {
       if (DDfetch(*svIt,v)) {
	   result.push_back(std::make_pair(it->second,it->first));
	   break;  
	 }	  
     }  
  }
  return result;
}

#endif
