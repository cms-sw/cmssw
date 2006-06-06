#ifndef FileReaderSPNewFormat_h
#define FileReaderSPNewFormat_h

#include <queue>
#include <IORawData/CSCTFCommissioning/src/SPReader.h>
using namespace std;

class FileReaderSPNewFormat : public SPReader
{
 public:
  FileReaderSPNewFormat();
  virtual ~FileReaderSPNewFormat() {}
  // DDUReader interface
  virtual void Configure();
  virtual int reset() {return 0;}
  virtual int enableBlock() {return 0;}
  virtual int disableBlock() {return 0;}
  virtual int endBlockRead() {return 0;}
  virtual int readSP(unsigned short **buf, const bool debug = false);
  
 protected:
  virtual int chunkSize();
  
  bool hasTrailer(const unsigned short*, unsigned length);
  bool isTrailer1(const unsigned short*);
  bool isTrailer2(const unsigned short*);

  unsigned findTrailer1(const unsigned short*, unsigned length);
  unsigned findTrailer2(const unsigned short*, unsigned length);

  unsigned pointer;

  unsigned short trailer[8];
  unsigned short header[8];
};

#endif
