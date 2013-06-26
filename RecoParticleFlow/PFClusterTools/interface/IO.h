#ifndef __IO__
#define __IO__

#include <string>
#include <vector>


#include <iostream>
#include <fstream>
// #include <strstream>
#include <sstream>
#include <iomanip>

#ifndef __CINT__
#include <fnmatch.h>
#endif


/// \brief General option file parser
///
/// \code
///  IO io("myfile.opt");
///  fECALthresh = 0.05;
///  io.GetOpt("ECAL", "threshold", fECALthresh);
/// \endcode
/// \author Colin Bernet
/// \date January 2002
namespace pftools {
class IO {
 private:
  /// all non empty, uncommented lines
  std::vector< std::pair< std::string, std::string> > fAllLines;

  /// parse one file
  bool ParseFile(const char* filename);
  
  /// counter
  int fCurline;
  
  /// current key 
  std::string fCurkey;
  
  /// current tag 
  std::string fCurtag;
 

 public:
  /// maximum line size
  static const unsigned sLinesize;

  /// builds IO from files matching filepattern 
  IO(const char* filepattern);
  ~IO() {fAllLines.clear();}

  /// dumps fAllLines
  void Dump(std::ostream& out = std::cout) const;

  /// dumps fAllLines
  friend std::ostream& operator<<(std::ostream& out, IO& io);

  /// true if constructor went wrong
  bool IsZombie() const {return !fAllLines.size();}

#ifndef __CINT__
  /// reads a vector of T
  template <class T>
    bool GetOpt(const char* tag, const char* key, std::vector< T >& values) const; 
  
  /// reads a T
  template <class T>
    bool GetOpt(const char* tag, const char* key, T& value) const; 

  /// reads a string
  bool GetOpt(const char* tag, const char* key, std::string& value) const;

  /// \brief reads a vector of T
  ///
  /// this function allows to read several lines with the same tag/key:
  /// \code
  /// while ( GetAllOpt(tag, key, values) ) 
  /// //    do something...
  /// \endcode
  template <class T>  
    bool GetAllOpt(const char* tag, const char* key, std::vector< T >& values);       

  /// \brief reads a T
  ///
  /// this function allows to read several lines with the same tag/key:
  ///
  /// \code
  /// while ( GetAllOpt(tag, key, value) ) 
  /// //    do something...
  /// \endcode
  template <class T>
    bool GetAllOpt(const char* tag, const char* key, T& value); 

  std::string GetLineData(const char* tag, const char* key) const;
  std::string GetNextLineData(const char* tag, const char* key);

  
#endif
};
}

#ifndef __CINT__

template <class T>
bool pftools::IO::GetOpt(const char* tag, const char* key, std::vector< T >& values) const {
  std::string data = GetLineData(tag,key);
 
  std::istringstream in(data.c_str());  
  while(1) {
    T tmp;
    in>>tmp;
    if(!in) break;
    values.push_back(tmp);
  } 

  return true;
}

template <class T>
bool pftools::IO::GetOpt(const char* tag, const char* key, T& value) const {
  std::string data = GetLineData(tag,key);
  
  std::istringstream in(data.c_str());  
  in>>value;
  if(in.good()) return true;
  else return false;
}

template <class T>  
bool pftools::IO::GetAllOpt(const char* tag, const char* key, std::vector< T >& values) {
  return false;
} 

template <class T>
bool pftools::IO::GetAllOpt(const char* tag, const char* key, T& value) {
  std::string data = GetNextLineData(tag, key);
  if(data.empty()) return false;
  
  std::istringstream in(data.c_str());  
  in>>value;
  if(in) {
    return true;
  }
  else {
    return false;  
  }
}


#endif // __CINT__

#endif









