#ifndef DataFormats_Provenance_FileFormatVersion_h
#define DataFormats_Provenance_FileFormatVersion_h

#include <iosfwd>

namespace edm 
{
  struct FileFormatVersion 
  {
    FileFormatVersion() : value_(-1) { }
    explicit FileFormatVersion(int vers) : value_(vers)  { }
    bool isValid() const { return value_ >= 0; }

    bool fastCopyPossible() const { return value_ >= 8; }
    
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

  std::ostream&
  operator<< (std::ostream& os, FileFormatVersion const& ff);

/*
  inline
  FileFormatVersion getFileFormatVersion() {
    static FileFormatVersion const fileFormatVersion;
    return fileFormatVersion; 
  }
*/
}
#endif
