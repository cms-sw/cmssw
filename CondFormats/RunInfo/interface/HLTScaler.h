#ifndef CondFormats_RunInfo_HLTScaler_h
#define CondFormats_RunInfo_HLTScaler_h
/** 
 * $Id: LuminosityInfo.h,v 1.1 2009/02/17 11:05:53 xiezhen Exp $
 *
 ************************************************************/
#include <vector>
//persistable class
namespace lumi{
  struct HLTInfo {
    HLTInfo(){}
    HLTInfo(int i,int a,int p):
      inputcount(i),acceptcount(a),prescalefactor(p){}
    int inputcount;
    int acceptcount;
    int prescalefactor;
  };
  static const HLTInfo HLTNULL=HLTInfo(0,0,0);  
  typedef std::vector<HLTInfo>::const_iterator HLTIterator;  
  class HLTScaler{
  public:
    HLTScaler();
    size_t nHLTtrigger()const;
    //sequential access to HLTInfo
    HLTIterator hltBegin()const;
    HLTIterator hltEnd()const;
    ///
    ///setter methods. 
    ///
    void setHLTData(const std::vector<HLTInfo>& hltdetail);
  private:
    std::vector<HLTInfo> m_hltinfo;
  }; 
}
#endif 
