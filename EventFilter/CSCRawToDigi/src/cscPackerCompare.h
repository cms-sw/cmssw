#include <iostream>
#include <typeinfo>
#include "EventFilter/CSCRawToDigi/src/bitset_append.h"


/** Compares two objects, and prints them out if they differ
 */

template <class T>
bool cscPackerCompare(const T & t1, const T & t2)
{
  bool result = true;
  if(t1 != t2) {
    std::cerr << "Mismatch:\n"<< t1 << "\n" << t2 << std::endl;
    result = false;
  }
  return result;
}


template <class T>
T cscPackAndUnpack(T & t)
{
  boost::dynamic_bitset<> firstPack = t.pack();
  unsigned char data[10000];
  bitset_utilities::bitsetToChar(firstPack, data);
  return T((unsigned short int *)data);
}


// packs a class, then unpacks, packs again, and compares
template <class T>
bool cscClassPackerCompare(T & t)
{
  boost::dynamic_bitset<> firstPack = t.pack();
  unsigned char data[1000];
  bitset_utilities::bitsetToChar(firstPack, data);
  T newObject((unsigned short int *)data);
  boost::dynamic_bitset<> secondPack = newObject.pack();
  if(firstPack != secondPack)
  {
    std::cerr << "Mismatch in " << typeid(t).name() << "\n";
    bitset_utilities::printWords(firstPack);
    bitset_utilities::printWords(secondPack);
    return false;
  }
  return true;
}
     
