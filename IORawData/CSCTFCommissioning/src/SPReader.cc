#include <iostream>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <cstdio>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <string>
#include <cstring>

#include <IORawData/CSCTFCommissioning/src/SPReader.h>

using namespace std;

int SPReader::openFile(string filename) {
  
  fd_schar = open(filename.c_str(), O_RDONLY);
  // Abort in case of any failure
  if (fd_schar == -1) {
    cerr << "SPReader: FATAL in openFile - " << strerror(errno) << endl;
    cerr << "SPReader will abort!!!" << endl;
    abort();
  }
  return 0;
}

bool SPReader::readNextEvent() {
  bool debug = false;
  if(theBuffer)
    {
      delete theBuffer;
      theBuffer = NULL;
    }
  unsigned short ** buf2 = new unsigned short*;
  theDataLength = readSP(buf2, debug);
  if (debug) cout << " theDataLength " << theDataLength << endl;
  if(theDataLength<=4) return false;
  unsigned short * buf=(unsigned short *)*buf2;
  theBuffer = buf;
  delete buf2;
  return true;
}

void SPReader::printStats()
{
  cout << " npackets " << dec << npack_schar
       << " nbytes " << nbytes_schar << endl;
}

void SPReader::closeFile() {
  if(fd_schar != -1000)
    close(fd_schar);
}
