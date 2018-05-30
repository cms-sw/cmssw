#ifndef DQM_SiStripCommissioningSources_DaqScopeModeTask_h
#define DQM_SiStripCommissioningSources_DaqScopeModeTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
   @class DaqScopeModeTask
*/
class DaqScopeModeTask : public CommissioningTask {

 public:
  
  DaqScopeModeTask( DQMStore*, const FedChannelConnection&, const edm::ParameterSet & );
  ~DaqScopeModeTask() override;
  
 private:

  void book() override;
  void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& ) override;

  void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>&,
		     const edm::DetSet<SiStripRawDigi>&) override;

  void update() override;

  // scope mode frame for each channel
  HistoSet scopeFrame_;

  // Pedestal and common mode
  std::vector<HistoSet> peds_;
  std::vector<HistoSet> cm_;

  uint16_t nBins_;
  uint16_t nBinsSpy_;

  /// parameters useful for the spy
  edm::ParameterSet parameters_;
};

#endif // DQM_SiStripCommissioningSources_DaqScopeModeTask_h

