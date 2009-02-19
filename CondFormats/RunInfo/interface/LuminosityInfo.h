#ifndef CondFormats_RunInfo_LuminosityInfo_h
#define CondFormats_RunInfo_LuminosityInfo_h
/** 
 *
 * BunchCrossingInfo holds Details information the lumi value, the error on this value 
 * and its quality for each bunch crossing (BX) in a given luminosity section (LS)   
 * BX definition: There are 3564 bunch crossing (BX) in each LHC orbit 
 * each event will occur at one of these BX. BX is defined to be the number of the
 * bunch crossing where this event occurred.
 *
 * $Id: LuminosityInfo.h,v 1.1 2009/02/17 11:05:53 xiezhen Exp $
 *
 ************************************************************/
 
#include <vector>

namespace lumi{
  static const int BXMIN=1;
  static const int BXMAX=3564;

  static const int LUMIALGOMAX=3;
  typedef enum { ET=0,OCCD1=1,OCCD2=2} LumiAlgoType;

  //persistable class
  struct BunchCrossingInfo {
    BunchCrossingInfo(){}
    BunchCrossingInfo(int idx,float value,float err,int quality,int norm):
      BXIdx(idx),lumivalue(value),lumierr(err),lumiquality(quality),normalization(norm){}
    int BXIdx;//starting from 1
    float lumivalue; 
    float lumierr;
    int lumiquality;
    int normalization;
  };
  static const BunchCrossingInfo BXNULL=BunchCrossingInfo(0,0.0,0.0,0,0);
  typedef std::vector<BunchCrossingInfo>::const_iterator BunchCrossingIterator;

  //persistable class
  struct LumiAverage{
    LumiAverage(){}
    LumiAverage(float v,float e, int q,int norm):value(v),error(e),quality(q),normalization(norm){}
    float value;
    float error;
    int   quality;
    int normalization;
  };
  
  //main persistable class
  class LuminosityInfo{
  public:
    LuminosityInfo();
    ~LuminosityInfo(){}
  public:
    ///
    ///getter methods
    ///
    int lumisectionID()const;
    float deadTimeNormalization()const;
    size_t nBunchCrossing()const;
    //radom access to LumiAverage by algorithm
    LumiAverage lumiAverage( const LumiAlgoType lumialgotype )const;
    //random access to bunchCrossingInfo by index
    const BunchCrossingInfo bunchCrossingInfo( const int BXIndex,
					 const LumiAlgoType lumialgotype )const;
    //sequential access to bunchCrossingInfo
    BunchCrossingIterator bunchCrossingBegin( const LumiAlgoType lumialgotype )const;
    BunchCrossingIterator bunchCrossingEnd( const LumiAlgoType lumialgotype )const;
    ///
    ///setter methods. 
    ///
    void setLumiSectionId(int sectionid);
    void setDeadtimeNormalization(float dtimenorm);
    void setLumiAverage(const LumiAverage& avg,
			const LumiAlgoType algotype);
    void setBunchCrossingData(const std::vector<BunchCrossingInfo>& BXs,
			      const LumiAlgoType algotype);
  private:
    std::vector<BunchCrossingInfo> m_bx;
    int m_sectionid; 
    float m_deadtime_normalization;
    std::vector<LumiAverage> m_summaryinfo;
  }; 
}//ns lumi
#endif 
