//_______________________________________
//
//  Class for TMB Logic Analyzer Data
//  CSCTMBScope 9/11/03  B.Mohr
//_______________________________________
//

#ifndef EventFilter_CSCRawToDigi_CSCTMBScope_h
#define EventFilter_CSCRawToDigi_CSCTMBScope_h

#ifndef LOCAL_UNPACK
#include <atomic>
#endif

class CSCTMBScope {
public:
  CSCTMBScope() { size_ = 0; }  //default constructor
  CSCTMBScope(const uint16_t *buf, int b05Line, int e05Line);
  static unsigned short sizeInWords() { return 1538; }
  static void setDebug(const bool value) { debug = value; };

  unsigned int data[52];  //scope data for ntuple
                          //public for now -- better way?
private:
  int UnpackScope(const uint16_t *buf, int b05Line, int e05Line);
  int GetPretrig(int ich);

  unsigned int scope_ram[256][6];  //stores all scope data
  unsigned short size_;
#ifdef LOCAL_UNPACK
  static bool debug;
#else
  static std::atomic<bool> debug;
#endif
};

#endif
