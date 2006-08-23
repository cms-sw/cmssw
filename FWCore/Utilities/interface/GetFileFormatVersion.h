#ifndef Utilities_GetFileFormatVersion_h
#define Utilities_GetFileFormatVersion_h

namespace edm 
{
  // We do not inline this function to help avoid inconsistent
  // versions being inlined into different libraries.
  
  int getFileFormatVersion();
}
#endif
