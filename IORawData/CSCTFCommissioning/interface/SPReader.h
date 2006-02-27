#ifndef SPReader_h
#define SPReader_h

#include <fstream>
#include <string>

using namespace std;

class SPReader
{
 public:
  SPReader() : nbytes_schar(0), npack_schar(0), nbyte_schar(0), theBuffer(NULL) {}
  virtual ~SPReader() {};
  virtual void Configure() {};
  virtual void Enable() {};
  /// assumes fd_schar has been set
  virtual int readSP(unsigned short **buf, const bool debug = false) = 0;
  /// assumes readNextEvent has been called
  bool readNextEvent();

  virtual char * data() {return (char *) theBuffer;}
  virtual int dataLength() {return theDataLength;}
  virtual int reset()        = 0;
  virtual int enableBlock()  = 0;
  virtual int disableBlock() = 0;
  virtual int endBlockRead() = 0;
  
  virtual void printStats();
  int openFile(string filename);
  void closeFile();
  
  unsigned short errorFlag;
   

 protected:
  /// How many bytes to read at a time
  virtual int chunkSize() = 0;

  int fd_schar;
  unsigned long nbytes_schar;
  int npack_schar;
  int nbyte_schar;
  bool liveData_;

  //buffer containing event data
  unsigned short * theBuffer;
  int theDataLength;

};

#endif
