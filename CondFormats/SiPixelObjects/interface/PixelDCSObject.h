#ifndef CondFormats_SiPixelObjects_PixelDCSObject_h
#define CondFormats_SiPixelObjects_PixelDCSObject_h

/** \class PixelDCSObject
 *
 *  Template struct for Pixel DCS object.
 *
 *  Value type is specified by the template parameter Type.
 *  Define a new struct for non-POD value type.
 *
 *  $Date: 2008/11/30 19:41:08 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

#include <map>
#include <string>

#include "FWCore/Utilities/interface/Exception.h"

template <class T>
class PixelDCSObject
{
  public:

  typedef T Type;

  typedef typename std::map<std::string, Type> Map;
  typedef typename Map::value_type Item;

  Type& operator [](const std::string& name) { return items[name]; }

  const Type& getValue(const std::string& name) const;

  private:

  Map items;
};

template <class Type>
const Type& PixelDCSObject<Type>::getValue(const std::string& name) const
{
  typename Map::const_iterator f = items.find(name);

  if (items.end() == f)
  {
    throw cms::Exception("PixelDCSObject")
        << "Cannot find item for " << name << " in DCS object.";
  }

  return f->second;
}

struct CaenChannel
{
  bool isOn;  // true if channel is on
  float iMon; // current value
  float vMon; // voltage value
};

#endif
