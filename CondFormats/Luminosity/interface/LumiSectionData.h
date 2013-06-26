#ifndef CondFormats_Luminosity_LumiSectionData_h
#define CondFormats_Luminosity_LumiSectionData_h
/** 
 *
 * BunchCrossingInfo holds Details information the lumi value, the error on this value 
 * and its quality for each bunch crossing (BX) in a given luminosity section (LS)   
 * BX definition: There are 3564 bunch crossing (BX) in each LHC orbit 
 * each event will occur at one of these BX. BX is defined to be the number of the
 * bunch crossing where this event occurred.
 *
 * $Id: LumiSectionData.h,v 1.3 2009/12/04 14:09:11 xiezhen Exp $
 *
 ************************************************************/
 
#include <vector>
#include <string>

namespace lumi{
  static const int BXMIN=1;
  static const int BXMAX=3564;
  static const int LUMIALGOMAX=3;
  
  typedef enum { ET=0,OCCD1=1,OCCD2=2 } LumiAlgoType;
  typedef enum { ALGO=0,TECH=1 } TriggerType;

  struct HLTInfo{
    HLTInfo():pathname(""),inputcount(-99),acceptcount(-99),prescale(-99){}
    HLTInfo(const std::string& pathnameIN, int i,int a,int p):
      pathname(pathnameIN),inputcount(i),acceptcount(a),prescale(p){}
    std::string pathname;
    int inputcount;
    int acceptcount;
    int prescale;
  };

  struct TriggerInfo{
    TriggerInfo():name(""),triggercount(-99),deadtimecount(-99),prescale(-99){}
    TriggerInfo(const std::string& trgname,int trgcount,int deadcount,int p):name(trgname),triggercount(trgcount),deadtimecount(deadcount),prescale(p){}
    std::string name;
    int triggercount;
    int deadtimecount;//max 2**20*3564=3737124864, so wrong type
    int prescale; 
  };

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
  typedef std::vector< HLTInfo >::const_iterator HLTIterator;  
  typedef std::vector< TriggerInfo >::const_iterator TriggerIterator;

  class LumiSectionData{
  public:
    LumiSectionData();
    ~LumiSectionData(){}
  public:
    ///
    ///getter methods
    ///
    std::string lumiVersion()const;
    int lumisectionID()const;
    size_t nBunchCrossing()const;
    //radom access to instant LumiAverage 
    float lumiAverage()const;
    float lumiError()const;
    float deadFraction()const;
    int lumiquality()const;
    unsigned long long startorbit()const;
    //get bunchCrossingInfo by algorithm
    void bunchCrossingInfo(  const LumiAlgoType lumialgotype, 
			     std::vector<BunchCrossingInfo>& result )const ;
    //random access to bunchCrossingInfo by bunchcrossing index
    const BunchCrossingInfo bunchCrossingInfo( const int BXIndex,
					 const LumiAlgoType lumialgotype )const;
    //sequential access to bunchCrossingInfo
    BunchCrossingIterator bunchCrossingBegin( const LumiAlgoType lumialgotype )const;
    BunchCrossingIterator bunchCrossingEnd( const LumiAlgoType lumialgotype )const;
    //total number of HLT paths
    size_t nHLTPath()const;
    bool HLThasData()const;
    HLTIterator hltBegin()const;
    HLTIterator hltEnd()const;

    bool TriggerhasData()const;
    TriggerIterator trgBegin()const;
    TriggerIterator trgEnd()const;

    short qualityFlag()const;
    ///
    ///setter methods. 
    ///
    void setLumiNull(); //set versionid number to -99, signal no lumi data written.
    void setLumiVersion(const std::string& versionid);
    void setLumiSectionId(int sectionid);
    void setLumiAverage(float lumiavg);
    void setLumiQuality(int lumiquality);
    void setDeadFraction(float deadfrac);
    void setLumiError(float lumierr);
    void setStartOrbit(unsigned long long orbtnumber);
    void setBunchCrossingData(const std::vector<BunchCrossingInfo>& BXs,
			      const LumiAlgoType algotype);
    void setHLTData(const std::vector<HLTInfo>& hltdetail);
    void setTriggerData(const std::vector<TriggerInfo>& triggerinfo);
    void setQualityFlag(short qualityflag);
    void print( std::ostream& s )const;
  private:
    std::vector<BunchCrossingInfo> m_bx;//Lumi detail info sorted by algoright+BX number stored as blob
    int m_sectionid; //LS id counting from 1 as required by evt. Instead from 0
    std::string m_versionid; //Lumi version
    float m_lumiavg; //instant lumi , selected from best algo
    float m_lumierror; //instant lumi err, 
    short m_quality; //use 7 bits PIXEL,STRIP,MUON,HCAL,ECAL,HF,HLX
    float m_deadfrac; //deadtime fraction
    unsigned long long m_startorbit; //first orbit number of this LS    
    std::vector< HLTInfo > m_hlt; //hlt scaler information sorted by hltpath independent of lumiversion
    std::vector< TriggerInfo > m_trigger; //trigger scaler sorted by bit number 128algo+64tech independent of lumiversion
  }; 
}//ns lumi
#endif 
