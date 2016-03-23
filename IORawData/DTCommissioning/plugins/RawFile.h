#ifndef DaqSource_RawFile_h
#define DaqSource_RawFile_h

/** \class RawFile
 *  Utility class to open, read and manage local and rfio files
 *  in a transparent way
 *
 *  \author J. Alcaraz - CIEMAT, Madrid
 */

#include <iostream>
//#include <boost/cstdint.hpp>

class RawFile {
 public:
  /// Default constructor
  RawFile();

  /// Usual constructor
  RawFile(const char* path);

  /// Open file
  RawFile* open(const char* path);

  /// Close file if necessary
  int close();

  /// Destructor
  virtual ~RawFile();

  /// Get file pointer
  FILE* GetPointer();

  /// It is OK (i.e. file was correctly opened)
  bool ok();

  /// It is not OK
  bool fail();

  /// Castor flag
  bool isRFIO();

  /// XROOTD flag
  bool isXROOTD();

  /// Read from file
  int read(void* data, size_t nbytes);

  /// Go somewhere
  int seek(long offset, int whence);

  /// Ignore some bytes
  int ignore(long offset);

  /// Check end of file
  int eof();

  /// Tell instruction
  long tell();

 private:

  FILE* inputFile;
  bool  rfioFlag;
  bool  xrootdFlag;
};
#endif
