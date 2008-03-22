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
 * $Id: LumiSummary.h,v 1.1 2007/07/13 20:13:02 wdd Exp $
 *
 ************************************************************/
 
#include <vector>

class LumiSummary {
  public:

    /// default constructor
    LumiSummary():
      avginsdellumi_(-99.),
      avginsdellumierr_(-99.),
      lumisecqual_(-1),
      deadfrac_(-99.),
      lsnumber_(-1)
    { }

    /// set default constructor
    LumiSummary(float avginsdellumi, float avginsdellumierr,
	        int lumisecqual,
                float deadfrac, int lsnumber,
                const std::vector<int>& l1ratecounter, const std::vector<int>& l1scaler,
                const std::vector<int>& hltratecounter, const std::vector<int>& hltscaler,
                const std::vector<int>& hltinput):
      avginsdellumi_(avginsdellumi), avginsdellumierr_(avginsdellumierr), 
      lumisecqual_(lumisecqual),
      deadfrac_(deadfrac), lsnumber_(lsnumber),
      l1ratecounter_(l1ratecounter), l1scaler_(l1scaler),
      hltratecounter_(hltratecounter), hltscaler_(hltscaler),
      hltinput_(hltinput)
    { }

    /// destructor
    ~LumiSummary(){}
	 
    float avgInsDelLumi() const { return avginsdellumi_; }
  
    float avgInsDelLumiErr() const { return avginsdellumierr_; }
  
    int    lumiSecQual() const { return lumisecqual_; }

    float deadFrac() const { return deadfrac_; }
    float liveFrac() const { return (1.0f - deadfrac_); }
 
    int    lsNumber() const { return lsnumber_; }


    // other inline have to be made to return 
    // the rate counter and scalers based on the label

    int    l1RateCounter(int line) const {return l1ratecounter_.at(line);}
    int    l1Scaler(int line) const {return l1scaler_.at(line);}
    int    hltRateCounter(int line) const {return hltratecounter_.at(line);}
    int    hltScaler(int line) const {return hltscaler_.at(line);}
    int    hltInput(int line) const {return hltinput_.at(line);}

    const std::vector<int>&    l1RateCounter() const {return l1ratecounter_;}
    const std::vector<int>&    l1Scaler() const {return l1scaler_;}
    const std::vector<int>&    hltRateCounter() const {return hltratecounter_;}
    const std::vector<int>&    hltScaler() const {return hltscaler_;}
    const std::vector<int>&    hltInput() const {return hltinput_;}

    float avgInsRecLumi() const { return avginsdellumi_ * liveFrac(); }
  
    float avgInsRecLumiErr() const { return avginsdellumierr_ * liveFrac(); }
  
  private :

    float avginsdellumi_;
    float avginsdellumierr_;
    int    lumisecqual_;
    float deadfrac_;
    int    lsnumber_;

    std::vector<int> l1ratecounter_;
    std::vector<int> l1scaler_;
    std::vector<int> hltratecounter_;
    std::vector<int> hltscaler_;
    std::vector<int> hltinput_;
}; 

std::ostream& operator<<(std::ostream& s, const LumiSummary& lumiSummary);

#endif // DataFormats_Luminosity_LumiSummary_h
