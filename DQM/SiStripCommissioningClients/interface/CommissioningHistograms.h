#ifndef DQM_SiStripCommissioningClients_CommissioningHistograms_H
#define DQM_SiStripCommissioningClients_CommissioningHistograms_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummaryFactory.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "TProfile.h"
#include "TH1.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cstdint>

class CommissioningAnalysis;

class CommissioningHistograms {
public:
  // not used here but some uses in derived classes
  typedef dqm::harvesting::MonitorElement MonitorElement;
  typedef dqm::harvesting::DQMStore DQMStore;
  // ---------- con(de)structors ----------

  CommissioningHistograms(const edm::ParameterSet& pset, DQMStore* const, const sistrip::RunType&);

  // MAKE PRIVATE
  CommissioningHistograms();  // private constructor

  virtual ~CommissioningHistograms();

  virtual void configure(const edm::ParameterSet&, const edm::EventSetup&) {}

  // ---------- histogram container class ----------

  class Histo {
  public:
    Histo(const std::string& title, MonitorElement* const me, MonitorElement* const cme)
        : title_(title), me_(me), cme_(cme) {
      ;
    }
    Histo() : title_(""), me_(nullptr), cme_(nullptr) { ; }
    void print(std::stringstream&) const;
    std::string title_;
    MonitorElement* me_;
    MonitorElement* cme_;
  };

  // ---------- typedefs ----------

  typedef std::map<uint32_t, CommissioningAnalysis*> Analyses;

  typedef Analyses::iterator Analysis;

  typedef SummaryPlotFactory<CommissioningAnalysis*> Factory;

  typedef std::vector<Histo*> Histos;

  typedef std::map<uint32_t, Histos> HistosMap;

  typedef std::map<uint32_t, uint32_t> FedToFecMap;

  // ---------- histogram "actions" ----------

  static uint32_t runNumber(DQMStore* const, const std::vector<std::string>&);

  static sistrip::RunType runType(DQMStore* const, const std::vector<std::string>&);

  /** Extracts custom information from list of MonitorElements. */
  static void copyCustomInformation(DQMStore* const, const std::vector<std::string>&);

  void extractHistograms(const std::vector<std::string>&);

  // DEPRECATE
  void createCollations(const std::vector<std::string>&);

  virtual void histoAnalysis(bool debug);

  virtual void printAnalyses();

  virtual void printSummary();

  virtual void createSummaryHisto(const sistrip::Monitorable&,
                                  const sistrip::Presentation&,
                                  const std::string& top_level_dir,
                                  const sistrip::Granularity&);

  void remove(std::string pattern = "");

  void save(std::string& filename, uint32_t run_number = 0, std::string partitionName = "");

  // ---------- protected methods ----------

protected:
  inline const sistrip::RunType& task() const;

  inline DQMStore* const bei() const;

  Analyses& data(bool getMaskedData = false);

  inline Factory* const factory();

  inline const HistosMap& histos() const;

  inline const FedToFecMap& mapping() const;

  inline const edm::ParameterSet& pset() const;

  TH1* histogram(const sistrip::Monitorable&,
                 const sistrip::Presentation&,
                 const sistrip::View&,
                 const std::string& directory,
                 const uint32_t& xbins,
                 const float& xlow = 1. * sistrip::invalid_,
                 const float& xhigh = 1. * sistrip::invalid_);

  void printHistosMap();

  void clearHistosMap();

  // ---------- private member data ----------

protected:
  std::unique_ptr<Factory> factory_;

private:
  sistrip::RunType task_;

  DQMStore* bei_;

  Analyses data_;

  HistosMap histos_;

  FedToFecMap mapping_;

  edm::ParameterSet pset_;

  bool mask_;
  std::vector<uint32_t> fedMaskVector_;
  std::vector<uint32_t> fecMaskVector_;
  std::vector<uint32_t> ringVector_;
  std::vector<uint32_t> ccuVector_;
  std::vector<uint32_t> i2cChanVector_;
  std::vector<uint32_t> lldChanVector_;

  Analyses dataWithMask_;
  bool dataWithMaskCached_;
};

// ---------- inline methods ----------

const sistrip::RunType& CommissioningHistograms::task() const { return task_; }
CommissioningHistograms::DQMStore* const CommissioningHistograms::bei() const { return bei_; }
CommissioningHistograms::Factory* const CommissioningHistograms::factory() { return factory_.get(); }
const CommissioningHistograms::HistosMap& CommissioningHistograms::histos() const { return histos_; }
const CommissioningHistograms::FedToFecMap& CommissioningHistograms::mapping() const { return mapping_; }
const edm::ParameterSet& CommissioningHistograms::pset() const { return pset_; }

#endif  // DQM_SiStripCommissioningClients_CommissioningHistograms_H
