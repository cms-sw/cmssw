/** \file
 *
 *  \author J. Alcaraz
 */

#include <IORawData/DTCommissioning/plugins/RawFile.h>
#include <cstring>
#include <cstdio>

#include <XrdPosix/XrdPosixExtern.hh> 

using namespace std;

RawFile::RawFile() : inputFile(nullptr), xrootdFlag(false) {}

RawFile::RawFile(const char* path) : inputFile(nullptr), xrootdFlag(false) {
  open(path);
}

RawFile* RawFile::open(const char* path) {

  //cout << " Full path: " << path << endl;

  char* chaux = new char[strlen(path)+1];
  strcpy(chaux,path);
  char* prefix = strtok(chaux,":");
  //cout << " Prefix: " << prefix << endl;
  delete chaux;

  char* filename = prefix;
  if (strlen(prefix)<strlen(path)) filename = strtok(nullptr,":");
  //cout << " Filename: " << filename << endl;

  if (strcmp(prefix,"root")==0) xrootdFlag = true;

  if (xrootdFlag) {
    char chopt[] = "rb";
    inputFile = XrdPosix_Fopen(path,chopt);
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
  } else {
      flag = fclose(inputFile);
  }
  inputFile = nullptr;
  return flag;
}

RawFile::~RawFile(){close();}

FILE* RawFile::GetPointer(){ return inputFile;}

bool RawFile::ok(){ return (inputFile!=nullptr);}

bool RawFile::fail(){ return !ok();}

bool RawFile::isXROOTD() { return xrootdFlag;}

int RawFile::read(void* data, size_t nbytes) {
  if (xrootdFlag) {
    return XrdPosix_Fread(data,nbytes,1,inputFile);
  } else {
    return fread(data, nbytes, 1, inputFile);
  }
}

int RawFile::seek(long offset, int whence) {
  if (xrootdFlag) {
    return XrdPosix_Fseek(inputFile, offset, whence);
  } else {
    return fseek(inputFile, offset, whence);
  }
}

int RawFile::ignore(long offset) { return seek(offset, SEEK_CUR);}

int RawFile::eof() {
    return feof(inputFile);  // Also for XROOTD
}

long RawFile::tell() {
  if (xrootdFlag) {
    return XrdPosix_Ftell(inputFile);
  } else {
      return ftell(inputFile);
  }
}
