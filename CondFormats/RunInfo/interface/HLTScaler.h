#ifndef CondFormats_RunInfo_HLTScaler_h
#define CondFormats_RunInfo_HLTScaler_h
/** 
 * $Id: HLTScaler.h,v 1.1 2009/02/19 15:58:59 xiezhen Exp $
 *
 ************************************************************/
#include <vector>
#include <utility>
#include <string>
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
  static const HLTInfo HLTNULL=HLTInfo(-99,-99,-99);  
  typedef std::vector< std::pair<std::string,HLTInfo> >::const_iterator HLTIterator;  
  class HLTScaler{
  public:
    HLTScaler();
    //total number of HLT paths
    size_t nHLTPath()const;
    //sequential access to HLTInfo
    HLTIterator hltBegin()const;
    HLTIterator hltEnd()const;
    //get HLT info for a given path
    HLTInfo getHLTInfo( const std::string& pathname )const;
    ///
    ///setter methods. 
    ///
    void setHLTData(const std::vector< std::pair<std::string,HLTInfo> >& hltdetail);
  private:
    std::vector< std::pair<std::string,HLTInfo> > m_hltinfo;
  }; 
}
#endif 
