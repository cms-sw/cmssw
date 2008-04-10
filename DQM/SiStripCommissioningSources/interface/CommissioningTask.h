#ifndef DQM_SiStripCommissioningSources_CommissioningTask_H
#define DQM_SiStripCommissioningSources_CommissioningTask_H

#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiStripCommon/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "boost/cstdint.hpp"
#include <vector>
#include <string>
#include <iomanip>

class DQMStore;
class MonitorElement;

/**
   @class CommissioningTask
*/
class CommissioningTask {

 public: 
  
  // ---------- Constructors, destructors ----------

  CommissioningTask( DQMStore*, 
		     const FedChannelConnection&,
		     const std::string& my_name );

  virtual ~CommissioningTask();

  // ---------- Classes, structs ----------
  
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
    std::vector<float> vNumOfEntries_;
    std::vector<float> vSumOfContents_;
    std::vector<double> vSumOfSquares_;
    MonitorElement* histo_;
    bool isProfile_;
  };
  
  // ---------- Public methods ----------

  /** Books histograms and constructs HistoSet cache. */
  void bookHistograms();

  /** Fills HistoSet cache. */
  void fillHistograms( const SiStripEventSummary&, 
		       const edm::DetSet<SiStripRawDigi>& );
  
  /** Fill HistoSet cache for FED cabling (special case). */
  void fillHistograms( const SiStripEventSummary&, 
		       const uint16_t& fed_id,
		       const std::map<uint16_t,float>& fed_ch );
  
  /** Updates histograms using HistoSet cache. */
  void updateHistograms();

  /** Get histogram filled counter. */
  inline const uint32_t& fillCntr() const;
  
  /** Get histogram update frequency. */
  inline const uint32_t& updateFreq() const;

  /** Set histogram update frequency. */
  inline void updateFreq( const uint32_t& );
  
  /** Returns the name of this commissioning task. */
  inline const std::string& myName() const;
  
 protected: 
  
  // ---------- Protected methods ----------
  
  /** Updates the vectors of HistoSet. */
  void updateHistoSet( HistoSet&, const uint32_t& bin, const float& value );

  /** Updates the vectors of HistoSet. */
  void updateHistoSet( HistoSet&, const uint32_t& bin );

  /** Updates the MonitorElements of HistoSet. */
  void updateHistoSet( HistoSet& );
  
  /** Returns const pointer to DQM back-end interface object. */
  inline DQMStore* const dqm() const;

  /** */
  inline const FedChannelConnection& connection() const;
  
  /** Returns FEC key. */
  inline const uint32_t& fecKey() const;

  /** Returns FED key. */
  inline const uint32_t& fedKey() const;
  
 private: 
  
  // ---------- Private methods ----------

  CommissioningTask() {;}
  
  virtual void book();

  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );

  virtual void fill( const SiStripEventSummary&, 
		     const uint16_t& fed_id,
		     const std::map<uint16_t,float>& fed_ch );

  virtual void update();
  
  // ---------- Private member data ----------

  DQMStore* dqm_;

  uint32_t updateFreq_;

  uint32_t fillCntr_;

  FedChannelConnection connection_;

  uint32_t fedKey_;

  uint32_t fecKey_;

  bool booked_;

  std::string myName_;
  
};

// ----- inline methods -----

const uint32_t& CommissioningTask::fillCntr() const { return fillCntr_; }
const uint32_t& CommissioningTask::updateFreq() const { return updateFreq_; }
void CommissioningTask::updateFreq( const uint32_t& freq ) { updateFreq_ = freq; }
const std::string& CommissioningTask::myName() const { return myName_; }

DQMStore* const CommissioningTask::dqm() const { return dqm_; }
const FedChannelConnection& CommissioningTask::connection() const { return connection_; }

const uint32_t& CommissioningTask::fecKey() const { return fecKey_; }
const uint32_t& CommissioningTask::fedKey() const { return fedKey_; }

#endif // DQM_SiStripCommissioningSources_CommissioningTask_H

