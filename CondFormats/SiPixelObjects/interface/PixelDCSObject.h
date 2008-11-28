#ifndef CondFormats_SiPixelObjects_PixelDCSObject_h
#define CondFormats_SiPixelObjects_PixelDCSObject_h

#include <string>
#include <vector>

template <class Type>
struct PixelDCSObject
{
  struct Item
  {
    std::string name; // name of detector element

    Type value;
  };

  std::vector<Item> items;
};

#endif
