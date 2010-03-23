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
 *
 * \version   1st Version June 7 2007
 * $Id: LumiSummary.h,v 1.11 2010/03/22 18:58:08 xiezhen Exp $
 *
 ************************************************************/
 
#include <vector>
#include <iosfwd>
#include <string>
class LumiSummary {
 public:
  class L1{
  public:
    L1():triggername(""),ratecount(0),prescale(0){}
    std::string triggername;
    unsigned int ratecount;
    unsigned int prescale;
  };
  class HLT{
  public:
    HLT():pathname(""),ratecount(0),inputcount(0),prescale(0){}
    std::string pathname;
    unsigned int ratecount;
    unsigned int inputcount;
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
      lsnumber_(0),
      startorbit_(0),
      numorbit_(0)
    { 
      hltdata_.reserve(100);
      l1data_.reserve(128);
    }
    
    /// set default constructor
    LumiSummary(float avginsdellumi, 
		float avginsdellumierr,
	        short lumisecqual,
                unsigned long long deadcount, 
		int lsnumber,
                const std::vector<L1>& l1in,
		const std::vector<HLT>& hltin,
		unsigned int startorbit,
                unsigned int numorbit,
		const std::string& lumiversion):
      lumiversion_(lumiversion),
      avginsdellumi_(avginsdellumi), 
      avginsdellumierr_(avginsdellumierr), 
      lumisecqual_(lumisecqual),
      deadcount_(deadcount), lsnumber_(lsnumber),
      hltdata_(hltin), l1data_(l1in),
      startorbit_(startorbit),numorbit_(numorbit)
      { }

    /// destructor
    ~LumiSummary(){}
	 
    float avgInsDelLumi() const;
    float avgInsDelLumiErr() const;
    short lumiSecQual() const ;
    unsigned long long deadcount() const;
    float deadFrac() const ;
    float liveFrac() const;
    int lsNumber() const;
    unsigned int startOrbit() const;
    unsigned int numOrbit() const;
    bool isValid() const;

    // other inline have to be made to return 
    // the rate counter and scalers based on the label
    L1 l1info(unsigned int idx)const;
    L1 l1info(const std::string& name) const;
    
    HLT hltinfo(unsigned int idx)const;
    HLT hltinfo(const std::string& pathname) const;
    size_t nTriggerLine()const;
    size_t nHLTPath()const;
    std::vector<std::string> HLTPaths()const;
    float avgInsRecLumi() const;
    float avgInsRecLumiErr() const;
    bool isProductEqual(LumiSummary const& next) const;
    std::string lumiVersion()const;
  private :
    std::string lumiversion_;
    //instant lumi , selected from best algorithm
    float avginsdellumi_;
    //instant lumierror
    float avginsdellumierr_;
    //detector quality flag use HF,HLX    
    short lumisecqual_;
    unsigned long long deadcount_;
    int   lsnumber_;
    //contains about 100 - 200 hlt paths
    std::vector<HLT> hltdata_;
    //contains 128 + 64 triggers
    std::vector<L1> l1data_;
    //first orbit number of this LS
    unsigned int startorbit_;
    unsigned int numorbit_;
}; 

std::ostream& operator<<(std::ostream& s, const LumiSummary& lumiSummary);

#endif // DataFormats_Luminosity_LumiSummary_h
