#ifndef Alignment_CommonAlignment_AlignSetup_h
#define Alignment_CommonAlignment_AlignSetup_h

/** \class AlignSetup
 *
 *  A singleton class to hold objects used by modules in alignment.
 *
 *  Facilitate information transfer between different modules. An object
 *  created by one module can be easily accessed by another via AlignSetup.
 *
 *  AlignSetup has a template parameter to specify the type of objects it
 *  holds. Objects are stored in a map<string, Type*>. Users put/get an
 *  object by passing its name through the static methods put()/get().
 *  It returns 0 if the name is not found on get().
 *  It throws an exception if an object of the same name exists on put().
 *
 *  AlignSetup owns all the objects it holds. It deletes all the objects
 *  on destruction.
 *
 *  $Date: 2007/04/07 03:29:38 $
 *  $Revision: 1.4 $
 *  \author Chung Khim Lae
 */

#include <map>

#include "FWCore/Utilities/interface/Exception.h"

template <class Type>
class AlignSetup
{
  typedef typename std::map<std::string, Type> Container;

  public:

  /// Get an object from map using its name.
  /// A new object is default-constructed if the name does not exist.
  /// Can change object in the map.
  static Type& get(
		   const std::string& name = ""
		   );

  /// Find and return an object from map using its name.
  /// Throw an exception if the name does not exist.
  /// Cannot change object in the map.
  static const Type& find(
			  const std::string& name = ""
			  );

  private:

  /// Hide constructor.
  AlignSetup();

  static Container theStore;
};

template <class Type>
typename AlignSetup<Type>::Container AlignSetup<Type>::theStore;

template <class Type>
Type& AlignSetup<Type>::get(const std::string& name)
{
  return theStore[name];
}

template <class Type>
const Type& AlignSetup<Type>::find(const std::string& name)		     
{
  typename Container::const_iterator o = theStore.find(name);

  if (theStore.end() == o)
  {
    throw cms::Exception("AlignSetupError")
      << "Cannot find an object of name " << name << " in AlignSetup.";
  }

  return o->second;
}

#endif
