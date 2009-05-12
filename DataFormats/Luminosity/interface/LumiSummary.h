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
 * $Id: LumiSummary.h,v 1.6 2009/02/18 17:03:55 elmer Exp $
 *
 ************************************************************/
 
#include <vector>
#include <iosfwd>
#include <string>
class LumiSummary {
 public:
  class L1{
  public:
    L1():triggersource(""),ratecount(-99),scalingfactor(-99){}
    std::string triggersource;
    int ratecount;
    int scalingfactor;
  };
  class HLT{
  public:
    HLT():pathname(""),ratecount(-99),inputcount(-99),scalingfactor(-99){}
    std::string pathname;
    int ratecount;
    int inputcount;
    int scalingfactor;
  };

 public:
    /// default constructor
    LumiSummary():
      avginsdellumi_(-99.),
      avginsdellumierr_(-99.),
      lumisecqual_(-1),
      deadfrac_(-99.),
      lsnumber_(-1)
    { 
      hltdata_.reserve(100);
      l1data_.reserve(128);
    }
    
    /// set default constructor
    LumiSummary(float avginsdellumi, float avginsdellumierr,
	        int lumisecqual,
                float deadfrac, int lsnumber,
                const std::vector<L1>& l1in,
		const std::vector<HLT>& hltin):
      avginsdellumi_(avginsdellumi), avginsdellumierr_(avginsdellumierr), 
      lumisecqual_(lumisecqual),
      deadfrac_(deadfrac), lsnumber_(lsnumber),
      hltdata_(hltin), l1data_(l1in)
    { }

    /// destructor
    ~LumiSummary(){}
	 
    float avgInsDelLumi() const;
    float avgInsDelLumiErr() const;
    int   lumiSecQual() const ;
    float deadFrac() const ;
    float liveFrac() const;
    int lsNumber() const;
    bool isValid() const;

    // other inline have to be made to return 
    // the rate counter and scalers based on the label
    
    L1 l1info(int linenumber) const;
    std::string triggerConfig(int linenumber) const;
    HLT hltinfo(int idx)const;
    HLT hltinfo(const std::string& pathname) const;
    size_t nTriggerLine()const;
    size_t nHLTPath()const;
    std::vector<std::string> HLTPaths()const;
    float avgInsRecLumi() const;
    float avgInsRecLumiErr() const;
    bool isProductEqual(LumiSummary const& next) const;

  private :
    float avginsdellumi_;
    float avginsdellumierr_;
    int   lumisecqual_;
    float deadfrac_;
    int   lsnumber_;
    //contains about 100 hlt paths
    std::vector<HLT> hltdata_;
    //contains 128 triggers
    std::vector<L1> l1data_;
}; 

std::ostream& operator<<(std::ostream& s, const LumiSummary& lumiSummary);

#endif // DataFormats_Luminosity_LumiSummary_h
