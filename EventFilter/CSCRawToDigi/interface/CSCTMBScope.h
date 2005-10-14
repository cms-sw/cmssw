//_______________________________________
//
//  Class for TMB Logic Analyzer Data  
//  CSCTMBScope 9/11/03  B.Mohr           
//_______________________________________
//

#ifndef CSCTMBScope_h
#define CSCTMBScope_h

class CSCTMBScope {

public:

  CSCTMBScope() {size_ = 0;}  //default constructor
  CSCTMBScope(unsigned short *buf,int b05Line,int e05Line);
  static unsigned short sizeInWords() {return 1538;}
  static void setDebug(const bool value) {debug = value;};

  unsigned int data[52];            //scope data for ntuple
                                    //public for now -- better way?
private:

  int UnpackScope(unsigned short *buf,int b05Line,int e05Line);
  int GetPretrig(int ich);

  unsigned int scope_ram[256][6];   //stores all scope data
  unsigned short size_;
  static bool debug;

};

#endif

