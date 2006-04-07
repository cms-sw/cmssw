/// 04/07/06 A.Tumanov

#ifndef CSCDCCExaminer_h
#define CSCDCCExaminer_h


class CSCDCCExaminer {

public:
  CSCDCCExaminer(){}
  ~CSCDCCExaminer(){}

  static void setDebug(bool value) {debug = value;} 
 
  bool examine(unsigned short * buf);

  static bool debug;

private:
  int errorflag1;

};

#endif
