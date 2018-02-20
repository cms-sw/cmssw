#ifndef DQM_L1TMonitor_L1TStage2OMTF_h
#define DQM_L1TMonitor_L1TStage2OMTF_h

/*
 * \file L1TStage2OMTF.h
 * \Author Esmaeel Eskandari Tadavani
 * \December 2015 
*/

// system requirements
#include <iosfwd>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

// general requirements
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"

// stage2 requirements
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

// dqm requirements
#include "DQMServices/Core/interface/ConcurrentMonitorElement.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"

namespace omtfdqm {
  struct Histograms {
    ConcurrentMonitorElement omtf_hwEta;
    ConcurrentMonitorElement omtf_hwLocalPhi;
    ConcurrentMonitorElement omtf_hwPt;
    ConcurrentMonitorElement omtf_hwQual;
    ConcurrentMonitorElement omtf_proc;
    ConcurrentMonitorElement omtf_bx;

    ConcurrentMonitorElement omtf_hwEta_hwLocalPhi;
    ConcurrentMonitorElement omtf_hwPt_hwEta;
    ConcurrentMonitorElement omtf_hwPt_hwLocalPhi;

    ConcurrentMonitorElement omtf_hwEta_bx;
    ConcurrentMonitorElement omtf_hwLocalPhi_bx;
    ConcurrentMonitorElement omtf_hwPt_bx;
    ConcurrentMonitorElement omtf_hwQual_bx;
  };
}

// class decleration

class  L1TStage2OMTF: public DQMGlobalEDAnalyzer<omtfdqm::Histograms> {

public:

// class constructor
L1TStage2OMTF(const edm::ParameterSet & ps);
// class destructor
~L1TStage2OMTF() override;

// member functions
protected:
  void dqmAnalyze(const edm::Event&, const edm::EventSetup&, omtfdqm::Histograms const&) const override;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&, omtfdqm::Histograms&) const override;
  void bookHistograms(DQMStore::ConcurrentBooker&, const edm::Run&, const edm::EventSetup&, omtfdqm::Histograms&) const override ;

// data members
private:  

  std::string monitorDir;
  edm::InputTag omtfSource; 
  bool verbose ;
  edm::EDGetToken omtfToken; 
  float global_phi;

};

#endif
