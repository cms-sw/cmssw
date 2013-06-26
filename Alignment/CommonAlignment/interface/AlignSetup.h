#ifndef Alignment_CommonAlignment_AlignSetup_h
#define Alignment_CommonAlignment_AlignSetup_h

/** \class AlignSetup
 *
 *  A helper class to hold objects used by modules in alignment.
 *
 *  AlignSetup has a template parameter to specify the type of objects it
 *  holds. Objects are stored in a map<string, Type>. Users put/get an
 *  object by passing its name through the static methods put()/get().
 *  It returns 0 if the name is not found on get().
 *  It throws an exception if an object of the same name exists on put().
 *
 *  $Date: 2008/02/15 18:11:45 $
 *  $Revision: 1.4 $
 *  \author Chung Khim Lae
 */

#include <map>
#include <sstream>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

template <class Type>
class AlignSetup
{
  typedef typename std::map<std::string, Type> Container;

public:

  /// Constructor
  AlignSetup() {}

  /// Get an object from map using its name.
  /// A new object is default-constructed if the name does not exist.
  Type& get( const std::string& name = "" );

  /// Find and return an object from map using its name.
  /// Throw an exception if the name does not exist.
  Type& find( const std::string& name = "" );

  /// Print the name of all stored data
  void dump( void ) const;

private:
  Container theStore;
};


template <class Type>
Type& AlignSetup<Type>::get(const std::string& name)
{
  return theStore[name];
}

template <class Type>
Type& AlignSetup<Type>::find(const std::string& name)		     
{
  typename Container::iterator o = theStore.find(name);

  if (theStore.end() == o) {
      std::ostringstream knownKeys;
      for (typename Container::const_iterator it = theStore.begin(); it != theStore.end(); ++it) {
	knownKeys << (it != theStore.begin() ? ", " : "") << it->first;
      }
      throw cms::Exception("AlignSetupError")
        << "Cannot find an object of name " << name << " in AlignSetup, know only "
	<< knownKeys.str() << ".";
    }
  
  return o->second;
}

template <class Type>
void AlignSetup<Type>::dump( void ) const
{
  edm::LogInfo("AlignSetup") << "Printing out AlignSetup: ";
  for ( typename Container::const_iterator it = theStore.begin();
        it != theStore.end(); ++it ) {
    edm::LogVerbatim("AlignSetup") << it->first << std::endl;
  }
}

#endif
