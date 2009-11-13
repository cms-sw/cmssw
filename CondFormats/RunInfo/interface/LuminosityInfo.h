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
 * $Id: LuminosityInfo.h,v 1.7 2009/05/09 18:59:22 xiezhen Exp $
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
    BunchCrossingInfo(int idx,float value,float err,int quality):
      BXIdx(idx),lumivalue(value),lumierr(err),lumiquality(quality){}
    int BXIdx;//starting from 1
    float lumivalue; 
    float lumierr;
    int lumiquality;
  };
  static const BunchCrossingInfo BXNULL=BunchCrossingInfo(-99,-99.0,-99.0,-99);
  typedef std::vector<BunchCrossingInfo>::const_iterator BunchCrossingIterator;

  //main persistable class
  class LuminosityInfo{
  public:
    LuminosityInfo();
    ~LuminosityInfo(){}
  public:
    ///
    ///getter methods
    ///
    bool isNullData() const; //if there is no lumi data written
    short lumiVersionNumber()const;
    int lumisectionID()const;
    size_t nBunchCrossing()const;
    //radom access to instant LumiAverage 
    float lumiAverage()const;
    float lumiError()const;
    float deadFraction()const;
    int lumiquality()const;
    //get bunchCrossingInfo by algorithm
    void bunchCrossingInfo(  const LumiAlgoType lumialgotype, 
			     std::vector<BunchCrossingInfo>& result )const ;
    //random access to bunchCrossingInfo by bunchcrossing index
    const BunchCrossingInfo bunchCrossingInfo( const int BXIndex,
					 const LumiAlgoType lumialgotype )const;
    //sequential access to bunchCrossingInfo
    BunchCrossingIterator bunchCrossingBegin( const LumiAlgoType lumialgotype )const;
    BunchCrossingIterator bunchCrossingEnd( const LumiAlgoType lumialgotype )const;
    
    ///
    ///setter methods. 
    ///
    void setLumiNull(); //set versionid number to -99, signal no lumi data written.
    void setLumiVersionNumber(short versionid);
    void setLumiSectionId(int sectionid);
    void setLumiAverage(float lumiavg);
    void setLumiQuality(int lumiquality);
    void setDeadFraction(float deadfrac);
    void setLumiError(float lumierr);
    void setBunchCrossingData(const std::vector<BunchCrossingInfo>& BXs,
			      const LumiAlgoType algotype);
  private:
    std::vector<BunchCrossingInfo> m_bx;
    int m_sectionid; 
    short m_versionid;
    float m_lumiavg;
    float m_lumierror;
    int   m_quality;
    float m_deadfrac;
  }; 
}//ns lumi
#endif 
