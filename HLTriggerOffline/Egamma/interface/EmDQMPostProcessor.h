#ifndef HLTriggerOffline_Egamma_EmDQMPosProcessor_H
#define HLTriggerOffline_Egamma_EmDQMPosProcessor_H


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

class EmDQMPostProcessor : public DQMEDHarvester {
 public:
  EmDQMPostProcessor(const edm::ParameterSet& pset);
  ~EmDQMPostProcessor() {};

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;
  TProfile* dividehistos(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, const std::string& num, const std::string& denom, const std::string& out,const std::string& label, const std::string& titel= "");

 private:
  
  /** a replacement for the function TGraphAsymmErrors::Efficiency(..) used with earlier 
      versions of ROOT (this functionality has been moved to a separate class TEfficiency) */
  static void Efficiency(int passing, int total, double level, double &mode, double &lowerBound, double &upperBound);

  /** read from the configuration: if set to true, efficiencies 
      are calculated with respect to reconstructed objects (instead 
      of generated objects). This is e.g. a useful option when 
      running on data. */
  bool noPhiPlots;
  bool normalizeToReco;
  bool ignoreEmpty;

  /** convenience method to get a histogram but checks first
      whether the corresponding MonitorElement is non-null.
      @return null if the MonitorElement is null */
  TH1F *getHistogram(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, const std::string &histoPath);

  std::string subDir_;
  
  /** dataset with which these histograms were produced.
      This is set by a user parameter in the configuration
      file.

      It is just used for writing it to the DQM output file.
      Useful to remember with which dataset a histogram file
      was produced. This code does not do much with this information
      (apart from copying it to the output file) but 
      it can be used when generating reports.
  */
  std::string dataSet_;
};

#endif
