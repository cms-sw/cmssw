#ifndef FileReaderSP_h
#define FileReaderSP_h

#include <queue>
#include <IORawData/CSCTFCommissioning/interface/SPReader.h>
using namespace std;

class FileReaderSP : public SPReader
{
 public:
  FileReaderSP();
  virtual ~FileReaderSP() {}
  // DDUReader interface
  virtual void Configure();
  virtual int reset() {return 0;}
  virtual int enableBlock() {return 0;}
  virtual int disableBlock() {return 0;}
  virtual int endBlockRead() {return 0;}
  virtual int readSP(unsigned short **buf, const bool debug = false);
  
 protected:
  virtual int chunkSize();

  int fillMiniBuf();
  bool isHeader();
  static const int MAXBUFSIZE = 6;
  int pointer;
  unsigned short refBuf[MAXBUFSIZE];
  unsigned short key[3];
  queue<unsigned short> miniBuf;

};

#endif
