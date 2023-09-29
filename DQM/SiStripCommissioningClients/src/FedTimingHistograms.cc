#include "DQM/SiStripCommissioningClients/interface/FedTimingHistograms.h"
#include "CondFormats/SiStripObjects/interface/FedTimingAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DQM/SiStripCommissioningAnalysis/interface/FedTimingAlgorithm.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;

// -----------------------------------------------------------------------------
/** */
FedTimingHistograms::FedTimingHistograms(const edm::ParameterSet& pset, DQMStore* bei)
    : CommissioningHistograms(pset.getParameter<edm::ParameterSet>("FedTimingParameters"), bei, sistrip::FED_TIMING),
      factory_(new Factory),
      optimumSamplingPoint_(15.),
      minDelay_(sistrip::invalid_),
      maxDelay_(-1. * sistrip::invalid_),
      deviceWithMinDelay_(sistrip::invalid_),
      deviceWithMaxDelay_(sistrip::invalid_) {
  cout << "[" << __PRETTY_FUNCTION__ << "]"
       << " Created object for APV TIMING histograms" << endl;
}

// -----------------------------------------------------------------------------
/** */
FedTimingHistograms::~FedTimingHistograms() { cout << "[" << __PRETTY_FUNCTION__ << "]" << endl; }

// -----------------------------------------------------------------------------
/** */
void FedTimingHistograms::histoAnalysis(bool debug) {
  // Clear std::map holding analysis objects
  data_.clear();

  // Reset minimum / maximum delays
  float time_min = 1. * sistrip::invalid_;
  float time_max = -1. * sistrip::invalid_;
  uint32_t device_min = sistrip::invalid_;
  uint32_t device_max = sistrip::invalid_;

  //   // Iterate through std::map containing std::vectors of profile histograms
  //   CollationsMap::const_iterator iter = collations().begin();
  //   for ( ; iter != collations().end(); iter++ ) {

  //     // Check std::vector of histos is not empty (should be 2 histos)
  //     if ( iter->second.empty() ) {
  //       cerr << "[" << __PRETTY_FUNCTION__ << "]"
  // 	   << " Zero collation histograms found!" << endl;
  //       continue;
  //     }

  //     // Retrieve pointers to profile histos for this FED channel
  //     std::vector<TH1*> profs;
  //     Collations::const_iterator ihis = iter->second.begin();
  //     for ( ; ihis != iter->second.end(); ihis++ ) {
  // OBSOLETE!!!
  //       TProfile* prof = ExtractTObject<TProfile>().extract( mui()->get( ihis->first ) );
  //       if ( prof ) { profs.push_back(prof); }
  //     }

  //     // Perform histo analysis
  //     FedTimingAnalysis anal( iter->first );
  //     FedTimingAlgorithm algo( &anal );
  //     algo.analysis( profs );
  //     data_[iter->first] = anal;

  //     // Check tick height is valid
  //     if ( anal.height() < 100. ) {
  //       cerr << "[" << __PRETTY_FUNCTION__ << "]"
  // 	   << " Tick mark height too small: " << anal.height() << endl;
  //       continue;
  //     }

  //     // Check time of rising edge
  //     if ( anal.time() > sistrip::maximum_ ) { continue; }

  //     // Find maximum time
  //     if ( anal.time() > time_max ) {
  //       time_max = anal.time();
  //       device_max = iter->first;
  //     }

  //     // Find minimum time
  //     if ( anal.time() < time_min ) {
  //       time_min = anal.time();
  //       device_min = iter->first;
  //     }

  //   }

  //   cout << "[" << __PRETTY_FUNCTION__ << "]"
  //        << " Analyzed histograms for "
  //        << collations().size()
  //        << " FED channels" << endl;

  // Adjust maximum (and minimum) delay(s) to find optimum sampling point(s)
  if (time_max > sistrip::maximum_ || time_max < -1. * sistrip::maximum_) {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
         << " Unable to set maximum time! Found unexpected value: " << time_max << endl;
    return;
  }

  SiStripFecKey max(device_max);
  cout << " Device (FEC/slot/ring/CCU/module/channel) " << max.fecCrate() << "/" << max.fecSlot() << "/"
       << max.fecRing() << "/" << max.ccuAddr() << "/" << max.ccuChan() << "/"
       << " has maximum delay (rising edge) [ns]:" << time_max << endl;

  SiStripFecKey min(device_min);
  cout << " Device (FEC/slot/ring/CCU/module/channel): " << min.fecCrate() << "/" << min.fecSlot() << "/"
       << min.fecRing() << "/" << min.ccuAddr() << "/" << min.ccuChan() << "/"
       << " has minimum delay (rising edge) [ns]:" << time_min << endl;

  // Set maximum time for all analysis objects
  std::map<uint32_t, FedTimingAnalysis>::iterator ianal = data_.begin();
  for (; ianal != data_.end(); ianal++) {
    ianal->second.max(time_max);
    if (debug) {
      std::stringstream ss;
      ianal->second.print(ss);
      cout << ss.str() << endl;
    }
  }
}

// -----------------------------------------------------------------------------
/** */
void FedTimingHistograms::createSummaryHisto(const sistrip::Monitorable& histo,
                                             const sistrip::Presentation& type,
                                             const std::string& directory,
                                             const sistrip::Granularity& gran) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;

  // Check view
  sistrip::View view = SiStripEnumsAndStrings::view(directory);
  if (view == sistrip::UNKNOWN_VIEW) {
    return;
  }

  // Analyze histograms
  histoAnalysis(false);

  // Extract data to be histogrammed
  factory_->init(histo, type, view, directory, gran);
  uint32_t xbins = factory_->extract(data_);

  // Create summary histogram (if it doesn't already exist)
  TH1* summary = histogram(histo, type, view, directory, xbins);

  // Fill histogram with data
  factory_->fill(*summary);
}
