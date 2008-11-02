#ifndef DD_READMAPTYPE_H
#define DD_READMAPTYPE_H

// #include <iostream>
#include <string>
#include <map>

class DDException;

namespace dddDetails {
  void errorReadMapType(const std::string & key) const throw (DDException);
}

//! a std::map<std::string,YourType> that offers a const operator[key]; if key is not stored in the std::map, a DDException is thrown 
/** otherwise, the ReadMapType works the same as std::map<std::string,YourType> */
template<class V> class ReadMapType : public std::map<std::string,V>
{
 public:
  ReadMapType() : std::map<std::string,V>() {}

  const V & operator[](const std::string & key) const throw (DDException)
   { 
      typename std::map<std::string,V>::const_iterator it = this->find(key); 
      if (it == this->end()) errorReadMapType(key);
      return it->second;
   }
   

   V & operator[](const std::string & key)
   {
      //std::cout << "non-const-called" << std::endl;
      return std::map<std::string,V>::operator[](key);
   }
  
private:

};

#endif // DD_READMAPTYE_H
