#ifndef Common_FileFormatVersion_h
#define Common_FileFormatVersion_h

#include "FWCore/Utilities/interface/GetFileFormatVersion.h"

namespace edm {
  struct FileFormatVersion {
    explicit FileFormatVersion() : value_(getFileFormatVersion()) {}
    int value_;
  };
/*
  inline
  FileFormatVersion getFileFormatVersion() {
    static FileFormatVersion const fileFormatVersion;
    return fileFormatVersion; 
  }
*/
}
#endif
