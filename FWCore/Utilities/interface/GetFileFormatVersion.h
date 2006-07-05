#ifndef Utilities_GetFileFormatVersion_h
#define Utilities_GetFileFormatVersion_h

namespace edm {
  inline
  int getFileFormatVersion() {
    // zero for now.
    static int const fileFormatVersion = 0;
    return fileFormatVersion; 
  }
}
#endif
