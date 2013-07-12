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

#include <string>
#include <vector>

template <class T>
struct PixelDCSObject
{
  typedef T Type;

  struct Item
  {
    std::string name; // name of detector element

    Type value;
  };

  std::vector<Item> items;
};

struct CaenChannel
{
  bool isOn;  // true if channel is on
  float iMon; // current value
  float vMon; // voltage value
};

#endif
