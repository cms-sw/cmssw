#ifndef CondFormats_RunInfo_HLTScaler_h
#define CondFormats_RunInfo_HLTScaler_h
/** 
 * $Id: HLTScaler.h,v 1.2 2009/05/08 17:32:55 xiezhen Exp $
 *
 ************************************************************/
#include <vector>
#include <utility>
#include <string>
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"

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
  static const HLTInfo HLTInfoNULL=HLTInfo(-99,-99,-99);
  typedef std::vector< std::pair<std::string,HLTInfo> >::const_iterator HLTIterator;  
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
    //get HLT info for a given path
    HLTInfo getHLTInfo( const std::string& pathname )const;
    bool isNullData()const;
    ///
    ///setter methods. 
    ///
    
    void setHLTNULL();
    //first lumisecion id=-99 signals there are no data taken for the entire run
    void setHLTData(edm::LuminosityBlockID lumiid, 
		    const std::vector< std::pair<std::string,HLTInfo> >& hltdetail);
  private:
    //current run
    int m_run;
    //current lumi section number
    int m_lsnumber;
    //hltinfo by HLTPATH
    std::vector< std::pair<std::string,HLTInfo> > m_hltinfo;
  }; 
}
#endif 
