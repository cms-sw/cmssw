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
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

// class decleration

class L1TStage2OMTF : public DQMEDAnalyzer {
public:
  // class constructor
  L1TStage2OMTF(const edm::ParameterSet& ps);
  // class destructor
  ~L1TStage2OMTF() override;

  // member functions
protected:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;

  // data members
private:
  std::string monitorDir;
  edm::InputTag omtfSource;
  bool verbose;
  edm::EDGetToken omtfToken;
  float global_phi;

  MonitorElement* omtf_hwEta;
  MonitorElement* omtf_hwLocalPhi;
  MonitorElement* omtf_hwPt;
  MonitorElement* omtf_hwQual;
  MonitorElement* omtf_proc;
  MonitorElement* omtf_bx;

  MonitorElement* omtf_hwEta_hwLocalPhi;
  MonitorElement* omtf_hwPt_hwEta;
  MonitorElement* omtf_hwPt_hwLocalPhi;

  MonitorElement* omtf_hwEta_bx;
  MonitorElement* omtf_hwLocalPhi_bx;
  MonitorElement* omtf_hwPt_bx;
  MonitorElement* omtf_hwQual_bx;
};

#endif
