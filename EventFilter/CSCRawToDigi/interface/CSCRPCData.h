#ifndef EventFilter_CSCRawToDigi_CSCRPCData_h
#define EventFilter_CSCRawToDigi_CSCRPCData_h

#include <vector>
#ifndef LOCAL_UNPACK
#include <atomic>
#endif

class CSCRPCDigi;

class CSCRPCData {
public:
  /// default constructor
  CSCRPCData(int ntbins = 7);
  // length is in 16-bit words
  CSCRPCData(const unsigned short *b04buf, int length);

  std::vector<int> BXN() const;
  std::vector<CSCRPCDigi> digis() const;
  void add(const CSCRPCDigi &);
  int sizeInWords() { return size_; }
  int nTbins() { return ntbins_; }
  void Print() const;
  bool check() const { return theData[0] == 0x6b04 && theData[size_ - 1] == 0x6e04; }

  static void setDebug(bool debugValue) { debug = debugValue; }

private:
#ifdef LOCAL_UNPACK
  static bool debug;
#else
  static std::atomic<bool> debug;
#endif
  int ntbins_;
  int size_;
  unsigned short theData[2 * 4 * 32 + 2];
};

#endif
