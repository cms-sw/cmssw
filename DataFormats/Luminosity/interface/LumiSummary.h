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
 * $Id$
 *
 ************************************************************/
 
#include <vector>

class LumiSummary {
  public:

    /// default constructor
    LumiSummary():
      avginslumi_(-99.),
      avginslumierr_(-99.),
      lumisecqual_(-1),
      deadfrac_(-99.),
      lsnumber_(-1)
    { }

    /// set default constructor
    LumiSummary(double avginslumi, double avginslumierr,
	        int lumisecqual,
                double deadfrac, int lsnumber,
                const std::vector<int>& l1ratecounter, const std::vector<int>& l1scaler,
                const std::vector<int>& hltratecounter, const std::vector<int>& hltscaler,
                const std::vector<int>& hltinput):
      avginslumi_(avginslumi), avginslumierr_(avginslumierr), 
      lumisecqual_(lumisecqual),
      deadfrac_(deadfrac), lsnumber_(lsnumber),
      l1ratecounter_(l1ratecounter), l1scaler_(l1scaler),
      hltratecounter_(hltratecounter), hltscaler_(hltscaler),
      hltinput_(hltinput)
    { }

    /// destructor
    ~LumiSummary(){}
	 
    double avgInsLumi() const { return avginslumi_; }
  
    double avgInsLumiErr() const { return avginslumierr_; }
  
    int    lumiSecQual() const { return lumisecqual_; }

    double deadFrac() const { return deadfrac_; }
    double liveFrac() const { return (1.0 - deadfrac_); }
 
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

  private :

    double avginslumi_;
    double avginslumierr_;
    int    lumisecqual_;
    double deadfrac_;
    int    lsnumber_;

    std::vector<int> l1ratecounter_;
    std::vector<int> l1scaler_;
    std::vector<int> hltratecounter_;
    std::vector<int> hltscaler_;
    std::vector<int> hltinput_;
}; 

std::ostream& operator<<(std::ostream& s, const LumiSummary& lumiSummary);

#endif // DataFormats_Luminosity_LumiSummary_h
