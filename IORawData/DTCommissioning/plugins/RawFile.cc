/** \file
 *
 *  \author J. Alcaraz
 */

#include <IORawData/DTCommissioning/plugins/RawFile.h>
#include <cstring>
#include <cstdio>

#include <XrdPosix/XrdPosixExtern.hh> 

using namespace std;

extern "C" {
  extern FILE* rfio_fopen(char *path, char *mode);
  extern int   rfio_fread(void*, size_t, size_t, void*);
  extern int   rfio_fclose(FILE *fd);
  extern int   rfio_fseek(FILE *fp, long offset, int whence);
  extern int   rfio_feof(FILE *fp);
  extern long  rfio_ftell(FILE *fp);
}

RawFile::RawFile() : inputFile(0), rfioFlag(false), xrootdFlag(false) {}

RawFile::RawFile(const char* path) : inputFile(0), rfioFlag(false), xrootdFlag(false) {
  open(path);
}

RawFile* RawFile::open(const char* path) {

  //cout << " Full path: " << path << endl;

  char* chaux = new char[strlen(path)+1];
  strcpy(chaux,path);
  char* prefix = strtok(chaux,":");
  //cout << " Prefix: " << prefix << endl;

  char* filename = prefix;
  if (strlen(prefix)<strlen(path)) filename = strtok(0,":");
  //cout << " Filename: " << filename << endl;

  if (strcmp(prefix,"rfio")==0) rfioFlag = true;
  if (strcmp(prefix,"castor")==0) rfioFlag = true;
  if (strcmp(prefix,"root")==0) xrootdFlag = true;

  if (xrootdFlag) {
    char chopt[] = "rb";
    inputFile = XrdPosix_Fopen(path,chopt);
  } else if (rfioFlag) {
    char chopt[] = "r";
    inputFile = rfio_fopen(filename,chopt);
  } else {
    char chopt[] = "rb";
    inputFile = fopen(filename,chopt);
  }
  if( !inputFile ) {
      cout << "RawFile: the input file '" << path << "' is not present" << endl;
  } else {
      cout << "RawFile: DAQ file '" << path << "' was succesfully opened" << endl;
  }

  return this;

}

int RawFile::close() {
  int flag = -1;
  if (!inputFile) return flag;

  if (xrootdFlag) {
    flag = XrdPosix_Fclose(inputFile);
  }
  else if (rfioFlag) {
      flag = rfio_fclose(inputFile);
  } else {
      flag = fclose(inputFile);
  }
  inputFile = 0;
  return flag;
}

RawFile::~RawFile(){close();}

FILE* RawFile::GetPointer(){ return inputFile;}

bool RawFile::ok(){ return (inputFile!=0);}

bool RawFile::fail(){ return !ok();}

bool RawFile::isRFIO() { return rfioFlag;}

bool RawFile::isXROOTD() { return xrootdFlag;}

int RawFile::read(void* data, size_t nbytes) {
  if (xrootdFlag) {
    return XrdPosix_Fread(data,nbytes,1,inputFile);
  }
  else if (rfioFlag) {
    return rfio_fread(data, nbytes, 1, inputFile);
  } else {
    return fread(data, nbytes, 1, inputFile);
  }
}

int RawFile::seek(long offset, int whence) {
  if (xrootdFlag) {
    return XrdPosix_Fseek(inputFile, offset, whence);
  }
  else if (rfioFlag) {
    return rfio_fseek(inputFile, offset, whence);
  } else {
    return fseek(inputFile, offset, whence);
  }
}

int RawFile::ignore(long offset) { return seek(offset, SEEK_CUR);}

int RawFile::eof() {
  if (rfioFlag) {
      return rfio_feof(inputFile);
  } else {
    return feof(inputFile);  // Also for XROOTD
  }
}

long RawFile::tell() {
  if (xrootdFlag) {
    return XrdPosix_Ftell(inputFile);
  }
  else if (rfioFlag) {
      return rfio_ftell(inputFile);
  } else {
      return ftell(inputFile);
  }
}
