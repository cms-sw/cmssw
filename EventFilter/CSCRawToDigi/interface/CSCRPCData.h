#ifndef CSCRPCData_h
#define CSCRPCData_h

#include <vector>

class CSCRPCDigi;


class CSCRPCData {
public:
  /// default constructor
  CSCRPCData(int ntbins=7);
  // length is in 16-bit words
  CSCRPCData(const unsigned short *b04buf , int length);

  std::vector<int> BXN() const;
  std::vector<CSCRPCDigi> digis() const;
  void add(const CSCRPCDigi &);
  int sizeInWords() {return size_;}
  int nTbins() {return ntbins_;}
  void Print() const;
  bool check() const {return theData[0]==0x6b04 && theData[size_-1] == 0x6e04;}

  static void setDebug(bool debugValue) {debug = debugValue;}
  
private:
  static bool debug;
  int ntbins_;
  int size_;
  unsigned short theData[2*4*32+2];
};

#endif

