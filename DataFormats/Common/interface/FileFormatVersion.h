#ifndef Common_FileFormatVersion_h
#define Common_FileFormatVersion_h

#include <ostream>

namespace edm 
{
  struct FileFormatVersion 
  {
    FileFormatVersion() : value_(-1) { }
    explicit FileFormatVersion(int vers) : value_(vers)  { }
    bool isValid() { return value_ >= 0; }
    
    int value_;
  };

  inline
  bool operator== (FileFormatVersion const& a, FileFormatVersion const& b)
  {
    return a.value_ == b.value_;
  }

  inline
  bool operator!= (FileFormatVersion const& a, FileFormatVersion const& b)
  {
    return !(a==b);
  }

  inline
  std::ostream&
  operator<< (std::ostream& os, FileFormatVersion const& ff)
  {
    os << ff.value_;
    return os;
  }

/*
  inline
  FileFormatVersion getFileFormatVersion() {
    static FileFormatVersion const fileFormatVersion;
    return fileFormatVersion; 
  }
*/
}
#endif
