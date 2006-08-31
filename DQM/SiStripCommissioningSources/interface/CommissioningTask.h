#ifndef DQM_SiStripCommissioningSources_CommissioningTask_H
#define DQM_SiStripCommissioningSources_CommissioningTask_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripEventSummary.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "DataFormats/SiStripDetId/interface/SiStripReadoutKey.h"
#include "boost/cstdint.hpp"
#include <string>
#include <iomanip>

class DaqMonitorBEInterface;
class MonitorElement;

using namespace std;

/**
   @class CommissioningTask
*/
class CommissioningTask {

 public: // ----- public interface -----

  /** Constructor. */ 
  CommissioningTask( DaqMonitorBEInterface*, 
		     const FedChannelConnection&,
		     const string& my_name );
  virtual ~CommissioningTask();


  /** Simple container class holding pointer to root histogram, and
      vectors in which data are cached and used to update histo. */
  class HistoSet {
  public:
    HistoSet() : 
      vNumOfEntries_(), 
      vSumOfContents_(), 
      vSumOfSquares_(), 
      histo_(0), 
      isProfile_(true) {;}
    // public data member
    vector<float> vNumOfEntries_;
    vector<float> vSumOfContents_;
    vector<double> vSumOfSquares_;
    MonitorElement* histo_;
    bool isProfile_;
  };
  
  void bookHistograms();
  void fillHistograms( const SiStripEventSummary&, 
		       const edm::DetSet<SiStripRawDigi>& );
  void fillHistograms( const SiStripEventSummary&, 
		       const uint16_t& fed_id,
		       const std::map<uint16_t,float>& fed_ch );
  
  void updateHistograms();
  
  /** Set histogram update frequency. */
  void updateFreq( const uint32_t& freq ) { updateFreq_ = freq; }
  
  /** Set FED id and channel (for FED cabling task). */
  //inline void fedChannel( const uint32_t& fed_key );
  
  /** Returns the name of this commissioning task. */
  const string& myName() const { return myName_; }
  
 protected: // ----- protected methods -----
  
  /** Updates the vectors of HistoSet. */
  void updateHistoSet( HistoSet&, const uint32_t& bin, const float& value );
  /** Updates the MonitorElements of HistoSet. */
  void updateHistoSet( HistoSet& );
  
  /** Returns const pointer to DQM back-end interface object. */
  inline DaqMonitorBEInterface* const dqm() const;

  /** */
  inline const FedChannelConnection& connection() const;
  
  /** Returns FEC key. */
  inline const uint32_t& fecKey() const;
  /** Returns FED key. */
  inline const uint32_t& fedKey() const;

  /** Returns FED id. */
  //inline const uint16_t& fedId() const;
  /** Returns FED channel. */
  //inline const uint16_t& fedCh() const;
  
 private: // ----- private methods -----
  
  CommissioningTask() {;}
  
  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void fill( const SiStripEventSummary&, 
		     const uint16_t& fed_id,
		     const std::map<uint16_t,float>& fed_ch );
  virtual void update();
  
 private: // ----- private data members -----

  DaqMonitorBEInterface* dqm_;
  uint32_t updateFreq_;
  uint32_t fillCntr_;
  FedChannelConnection connection_;
  uint32_t fedKey_;
  uint32_t fecKey_;
  bool booked_;
  //uint16_t fedId_;
  //uint16_t fedCh_;
  string myName_;
  
};

// ----- inline methods -----

DaqMonitorBEInterface* const CommissioningTask::dqm() const { return dqm_; }
const FedChannelConnection& CommissioningTask::connection() const { return connection_; }

const uint32_t& CommissioningTask::fecKey() const { return fecKey_; }
const uint32_t& CommissioningTask::fedKey() const { return fedKey_; }

//void CommissioningTask::fedChannel( const uint32_t& fed_key ) { 
//  SiStripReadoutKey::ReadoutPath path = SiStripReadoutKey::path( fed_key ); 
//  fedId_ = path.fedId_; fedCh_ = path.fedCh_;
//}

//const uint16_t& CommissioningTask::fedId() const { return fedId_; }
//const uint16_t& CommissioningTask::fedCh() const { return fedCh_; }

#endif // DQM_SiStripCommissioningSources_CommissioningTask_H

