//_______________________________________
//
//  Class for TMB Logic Analyzer Data
//  CSCTMBMiniScope   July 2010 Alexander Sakharov (Wayne State University)
//_______________________________________
//

#ifndef EventFilter_CSCRawToDigi_CSCTMBMiniScope_h
#define EventFilter_CSCRawToDigi_CSCTMBMiniScope_h
#include <vector>
#include <map>
#include <cstdint>

class CSCTMBMiniScope {
public:
  CSCTMBMiniScope() { size_ = 0; }  //default constructor
  CSCTMBMiniScope(const uint16_t *buf, int Line6b07, int Line6E07);
  int getSize() const { return size_; }
  int getTbinCount() const { return miniScopeTbinCount; }
  int getTbinPreTrigger() const { return miniScopeTbinPreTrigger; }
  std::vector<int> getAdr() const { return miniScopeAdress; }
  std::vector<int> getData() const { return miniScopeData; }
  std::vector<int> getChannelsInTbin(int data) const;

  void print() const;  /// Print the maped content of the miniscope

private:
  int UnpackMiniScope(const uint16_t *buf, int Line6b07, int Line6E07);

  std::vector<int> miniScopeAdress;  /// stores all mini scope adresses
  std::vector<int> miniScopeData;    /// stores all mini scope data
  int miniScopeTbinCount;
  int miniScopeTbinPreTrigger;
  unsigned size_;
};

#endif
