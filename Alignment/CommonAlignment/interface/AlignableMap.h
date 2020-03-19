#ifndef Alignment_CommonAlignment_AlignSetup_h
#define Alignment_CommonAlignment_AlignSetup_h

/** \class AlignableMap
 *
 *  A helper class to hold Alignables used by modules in alignment.
 *
 *  Alignables are stored in a map<string, Alignables>. Users get Alignables
 *  by passing the corresponding name through the method get(), if the name
 *  doesn't exist a new entry will be created. The find()-method also delivers
 *  Alignables per name, but it does not created new entries and will throw an
 *  error in case of an unknown name.
 *
 *  $Date: 2008/02/12 18:06:49 $
 *  $Revision: 1.3 $
 *  \author Chung Khim Lae
 *
 *  Last Update: Max Stark
 *         Date: Mon, 22 Feb 2016 19:58:45 CET
 */

#include <map>
#include <sstream>

#include "Alignment/CommonAlignment/interface/Utilities.h"

class AlignableMap {
  using Container = std::map<std::string, align::Alignables>;

public:
  AlignableMap(){};
  virtual ~AlignableMap(){};

  /// Get an object from map using its name.
  /// A new object is default-constructed if the name does not exist.
  align::Alignables& get(const std::string& name = "");

  /// Find and return an object from map using its name.
  /// Throw an exception if the name does not exist.
  align::Alignables& find(const std::string& name = "");

  /// Print the name of all stored data
  void dump(void) const;

private:
  Container theStore;
};

#endif
