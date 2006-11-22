#ifndef CSCTMBTrailer_h
#define CSCTMBTrailer_h

#include <string.h> // bzero

/** Defined to begin at the 6E0C word 
6E0C
2AAA (optional)
5555 (optional)
D8+CRC22(10)
D8+CRC22(10)
DE0F
D8+WordCount
*/

class CSCTMBTrailer {
public:
  /// don't forget to pass in the size of the tmb header + clct data
  CSCTMBTrailer(int wordCount=0);

  CSCTMBTrailer(unsigned short * buf);

  int crc22() const;
  bool check() const {return theData[0]==0x6e0c && theData[3+thePadding] == 0xde0f;}
  /// in 16-bit frames
  int sizeInWords() const {return 5+thePadding;}
  unsigned short * data() {return theData;}

  int wordCount() const;
  void setWordCount(int words);
private:
  unsigned short theData[7];
  int thePadding;
};

#endif

