#include "FWCore/Framework/interface/EDFilter.h"

class SiStripFedCabling;

class LaserAlignmentEventFilter : public edm::EDFilter {

 public:
  explicit LaserAlignmentEventFilter(const edm::ParameterSet&);
  ~LaserAlignmentEventFilter();

 private:

  // Name of the FED collection
  std::string FED_collection;

  // Map the std::vector<int> that is returned by edm::ParameterSet::getParameter() to internal representations (which are optimized for the Filter)
  void set_las_fed_ids(const std::vector<int>& las_feds);
  void set_las_signal_ids(const std::vector<int>& las_signal);
  
  virtual void beginRun(const edm::EventSetup&) ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // Filter Settings
  bool signal_filter; // Check for LAS Signals

  std::vector<uint16_t> las_fed_ids; // List of FEDs used by LAS
  std::vector<uint32_t> las_signal_ids; // List of DetIds to probe for signal

  uint16_t single_channel_thresh; // Signal threshold for a single channel
  uint16_t channel_count_thresh; // Nr. of channels that have to contain signal for LAS event
 
  // Debug output variables
  std::vector<uint16_t> buffer_size;
  std::vector<uint32_t> buffer_size_sum;
  uint32_t LAS_event_count;

  // Handling of FED Cabling
  void updateCabling( const edm::EventSetup& );
  const SiStripFedCabling* cabling;
  uint32_t cacheId_;
    
};
