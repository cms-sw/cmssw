/** \file
 *
 *  \author J. Alcaraz
 */

#include <IORawData/DTCommissioning/src/RawFile.h>
#include <cstring>
#include <cstdio>

using namespace std;

extern "C" {
  extern FILE* rfio_fopen(char *path, char *mode);
  extern int   rfio_fread(void*, size_t, size_t, void*);
  extern int   rfio_fclose(FILE *fd);
  extern int   rfio_fseek(FILE *fp, long offset, int whence);
  extern int   rfio_feof(FILE *fp);
  extern long  rfio_ftell(FILE *fp);
}
                                                                                
RawFile::RawFile() : inputFile(0), rfioFlag(false) {}

RawFile::RawFile(const char* path) : inputFile(0), rfioFlag(false) {
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

  if (rfioFlag) {
    char chopt[] = "r";
    inputFile = rfio_fopen(filename,chopt);
  } else {
    char chopt[] = "rb";
    inputFile = fopen(filename,chopt);
  }
  if( !inputFile ) {
      cout << "RawFile: the input file '" << filename << "' is not present" << endl;
  } else {
      cout << "RawFile: DAQ file '" << filename << "' was succesfully opened" << endl;
  }

  return this;

}

int RawFile::close() {
  int flag = -1;
  if (!inputFile) return flag;
  if (rfioFlag) {
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

int RawFile::read(void* data, size_t nbytes) {
      if (rfioFlag) {
            return rfio_fread(data, nbytes, 1, inputFile);
      } else {
            return fread(data, nbytes, 1, inputFile);
      }
}

int RawFile::seek(long offset, int whence) {
      if (rfioFlag) {
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
      return feof(inputFile);
  }
}

long RawFile::tell() {
  if (rfioFlag) {
      return rfio_ftell(inputFile);
  } else {
      return ftell(inputFile);
  }
}
