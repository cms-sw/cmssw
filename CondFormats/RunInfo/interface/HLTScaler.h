#ifndef CondFormats_RunInfo_HLTScaler_h
#define CondFormats_RunInfo_HLTScaler_h
/** 
 * $Id: HLTScaler.h,v 1.3 2009/05/11 17:33:55 xiezhen Exp $
 *
 ************************************************************/
#include <vector>
#include <utility>
#include <string>
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"

//persistable class
namespace lumi{
  struct HLTInfo {
    HLTInfo():pathname(""),inputcount(-99),acceptcount(-99),prescalefactor(-99){}
    HLTInfo(const std::string& pathnameIN, int i,int a,int p):
      pathname(pathnameIN),inputcount(i),acceptcount(a),prescalefactor(p){}
    std::string pathname;
    int inputcount;
    int acceptcount;
    int prescalefactor;
  };
  typedef std::vector< HLTInfo >::const_iterator HLTIterator;  
  class HLTScaler{
  public:
    HLTScaler();
    //current lumisection number
    int lumisectionNumber()const;
    //current run number
    int runNumber()const;
    //total number of HLT paths
    size_t nHLTPath()const;
    //sequential access to HLTInfo
    HLTIterator hltBegin()const;
    HLTIterator hltEnd()const;
    bool isNullData()const;
    ///
    ///setter methods. 
    ///
    
    void setHLTNULL();
    //first lumisecion id=-99 signals there are no data taken for the entire run
    void setHLTData(edm::LuminosityBlockID lumiid, 
		    const std::vector<HLTInfo>& hltdetail);
  private:
    //current run
    int m_run;
    //current lumi section number
    int m_lsnumber;
    //hltinfo by hltpath
    std::vector< HLTInfo > m_hltinfo;
  }; 
}
#endif 
