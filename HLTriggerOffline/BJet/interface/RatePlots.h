#ifndef HLTriggerOffline_BJet_RatePlots_h
#define HLTriggerOffline_BJet_RatePlots_h

// STL
#include <string>
#include <cmath>

// ROOT
#include <TDirectory.h>
#include <TH1F.h>

struct RatePlots {
  RatePlots() :
    m_rates(0)
  { }

  void init(const std::string & name, const std::string & title, unsigned int levels)
  {
    // enable sum-of-squares for all plots
    bool sumw2 = TH1::GetDefaultSumw2();
    TH1::SetDefaultSumw2(true);
    // disable directory association for all plots
    bool setdir = TH1::AddDirectoryStatus();
    TH1::AddDirectory(false);

    // a path with N filters can have N+1 rates: initial, after the 1st, ... after the Nth filter
    m_rates = new TH1I((name + "_rates").c_str(),  (title + " rates").c_str(), (levels+1), 0, (levels+1));

    // reset sum-of-squares status
    TH1::SetDefaultSumw2(sumw2);
    // reset directory association status
    TH1::AddDirectory(setdir);
  }
    
  // event passed given level 
  // level must be in [0..levels], with "0" meaning "no filter"
  void fill(unsigned int level)
  {
    m_rates->Fill(level);
  }

  // returns the number of events passing given level
  // level must be in [0..levels], with "0" meaning "no filter"
  unsigned int rate(unsigned int level) const
  {
    unsigned int value = 0;
    if (level > m_rates->GetNbinsX())
      value = 0;
    else
      // I *hate* the way ROOT handles bins - it's a capital offence by itself
      value = m_rates->GetBinContent( level+1 );
    return value;
  }
 
  // returns the differential efficiency of passing the given filter
  // level "0" means "no filter" and has efficiency 100% by default
  double stepEfficiency(unsigned int level) const
  {
    double value = 0.;
    if (level == 0)
      value = 1.;
    else if (level > m_rates->GetNbinsX())
      value = NAN;
    else if (m_rates->GetBinContent( level ) == 0)
      value = NAN;
    else
      value = (m_rates->GetBinContent( level+1 ) / m_rates->GetBinContent( level ));
    return value;
  }

  // returns the cumulative efficiency of passing the given filter, 
  // i.e. the efficiency w.r.t. level "0"
  double efficiency(unsigned int level) const
  {
    double value = 0.;
    if (level > m_rates->GetNbinsX())
      value = NAN;
    else if (m_rates->GetBinContent( 1 ) == 0)
      value = NAN;
    else
      value = (m_rates->GetBinContent( level+1 ) / m_rates->GetBinContent( 1 ));
    return value;
  }
  
  void save(TDirectory & file)
  {
    m_rates->SetDirectory(&file);
  }
  
  TH1 * m_rates;
};

#endif // HLTriggerOffline_BJet_RatePlots_h
