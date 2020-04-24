//_______________________________________
//
//  Class for TMB Logic Analyzer Data  
//  CSCTMBBlockedCFEB   July 2010 Alexander Sakharov (Wayne State University) 
//_______________________________________
//

#ifndef CSCTMBBlockedCFEB_h
#define CSCTMBBlockedCFEB_h
#include <vector>

class CSCTMBBlockedCFEB {

public:

  CSCTMBBlockedCFEB() {size_ = 0;}  //default constructor
  CSCTMBBlockedCFEB(unsigned short *buf,int Line6BCB,int Line6ECB);
  int getSize() const {return size_;}
  std::vector<int> getData() const {return BlockedCFEBdata;}
  std::vector< std::vector<int> > getSingleCFEBList(int CFEBn) const;

  void print() const;
  
private:

  int UnpackBlockedCFEB(unsigned short *buf,int Line6BCB,int Line6ECB);

  std::vector<int> BlockedCFEBdata;     /// stores all mini scope data
  unsigned size_;

};

#endif

