#include <iostream>

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

