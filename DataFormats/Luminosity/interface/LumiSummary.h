#ifndef DataFormats_Luminosity_LumiSummary_h
#define DataFormats_Luminosity_LumiSummary_h
 
/** \class LumiSummary
 *
 *
 * LumiSummary holds Summary information averaged over all bunch crossings
 * for the given luminosity section LS   
 *
 * \author Valerie Halyo
 *         David Dagenhart
 *         Zhen Xie
 * \version   1st Version June 7 2007
 * $Id: LumiSummary.h,v 1.20 2012/02/28 15:06:03 xiezhen Exp $
 *
 ************************************************************/
 
#include <vector>
#include <iosfwd>
#include <string>
class LumiSummary {
 public:
  class L1{
  public:
    L1():triggernameidx(-1),prescale(0){}
    int          triggernameidx;
    unsigned int prescale;
  };
  class HLT{
  public:
    HLT():pathnameidx(-1),prescale(0){}
    int          pathnameidx;
    unsigned int prescale;
  };

 public:
    /// default constructor
    LumiSummary():
      lumiversion_("-1"),
      avginsdellumi_(0.0),
      avginsdellumierr_(0.0),
      lumisecqual_(0),
      deadcount_(0),
      bitzerocount_(),
      lsnumber_(0),
      startorbit_(0),
      numorbit_(0)
    { 
      hltdata_.reserve(100);
      l1data_.reserve(192);
    }
    
    /// set default constructor
    LumiSummary(float avginsdellumi, 
		float avginsdellumierr,
	        short lumisecqual,
                unsigned long long deadcount, 
                unsigned long long bitzerocount, 
		unsigned int lsnumber,
                const std::vector<L1>& l1in,
		const std::vector<HLT>& hltin,
		unsigned int startorbit,
                unsigned int numorbit,
		const std::string& lumiversion):
      lumiversion_(lumiversion),
      avginsdellumi_(avginsdellumi), 
      avginsdellumierr_(avginsdellumierr), 
      lumisecqual_(lumisecqual),
      deadcount_(deadcount), bitzerocount_(bitzerocount), lsnumber_(lsnumber),
      hltdata_(hltin), l1data_(l1in),
      startorbit_(startorbit),numorbit_(numorbit)
      { }

    /// destructor
    ~LumiSummary(){}
    /** 
	average inst lumi,delivered, 
	unit Hz/ub, 
	uncalibrated for lumiversion v2 
    **/
    float avgInsDelLumi() const;
    /**
       average inst lumi error, empty for lumiversion v2 
    **/
    float avgInsDelLumiErr() const;
    /**
       delivered luminosity integrated over LS , 
       unit /ub,  
       uncalibrated for lumiversion v2 
    **/
    float intgDelLumi()const;
    short lumiSecQual() const ;
    /** 
	trigger DeadtimeBeamActive count 
    **/
    unsigned long long deadcount() const;
    /**
       trigger normalization bit count*prescale
    **/
    unsigned long long bitzerocount() const;
    /** the fraction trigger is not active=
	trigger DeadtimeBeamActive/BitZero
	special values:
	if trigger data absent for this LS, return deadfraction 1.0
	if bitzero=0 return -1.0 meaning no beam
    **/
    float deadFrac() const ;
    /** the fraction trigger is active=
    	1-deadfraction
    	special values:
    	if deadfraction<0(no beam) livefraction=0
	
     **/
    float liveFrac() const;
    /**lumi section length in seconds
       numorbits*3564*25e-09
    **/
    float lumiSectionLength() const;
    unsigned int lsNumber() const;
    unsigned int startOrbit() const;
    unsigned int numOrbit() const;
    /**data are valid only if run exists from all sources lumi,trg ,hlt
     **/
    bool isValid() const;
    //retrieve trigger bit by bit number 0-191(algo,tech)
    L1 l1info(unsigned int idx)const;
    HLT hltinfo(unsigned int idx)const;
    size_t nTriggerLine()const;
    size_t nHLTPath()const;
    /**avg inst lumi corrected by deadtime**/
    float avgInsRecLumi() const;
    /**avg inst lumi error corrected by deadtime**/
    float avgInsRecLumiErr() const;
    /**recorded luminosity integrated over LS **/
    float intgRecLumi()const;
    bool isProductEqual(LumiSummary const& next) const;
    /** lumi data version.
	special values:
	"-1" means not all lumi,trigger,hlt data exist, therefore invalid
     **/
    std::string lumiVersion()const;
    //
    //setters
    //
    void setLumiVersion(const std::string& lumiversion);
    void setLumiData(float instlumi,float instlumierr,short lumiquality);
    void setDeadCount(unsigned long long deadcount);
    void setBitZeroCount(unsigned long long bitzerocount);
    void setlsnumber(unsigned int lsnumber);
    void setOrbitData(unsigned int startorbit,unsigned int numorbit);
    void swapL1Data(std::vector<L1>& l1data);
    void swapHLTData(std::vector<HLT>& hltdata);
    void copyL1Data(const std::vector<L1>& l1data);
    void copyHLTData(const std::vector<HLT>& hltdata);

  private :
    std::string lumiversion_;
    //instant lumi , selected from best algorithm
    float avginsdellumi_;
    //instant lumierror
    float avginsdellumierr_;
    //detector quality flag use HF,HLX    
    short lumisecqual_;
    unsigned long long deadcount_;
    unsigned long long bitzerocount_;
    unsigned int lsnumber_;
    //contains about 100 - 200 hlt paths
    std::vector<HLT> hltdata_;
    //contains 128 + 64 triggers
    std::vector<L1> l1data_;
    //first orbit number of this LS
    unsigned int startorbit_;
    //number of orbits in this LS
    unsigned int numorbit_;
}; 

std::ostream& operator<<(std::ostream& s, const LumiSummary& lumiSummary);

#endif // DataFormats_Luminosity_LumiSummary_h
