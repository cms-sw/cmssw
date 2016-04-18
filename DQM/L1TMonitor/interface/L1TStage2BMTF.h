#ifndef DQM_L1TMonitor_L1TStage2BMTF_h
#define DQM_L1TMonitor_L1TStage2BMTF_h

/*
 * \file L1TStage2BMTF.h
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
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"


// class decleration

class  L1TStage2BMTF: public DQMEDAnalyzer {

public:

// class constructor
L1TStage2BMTF(const edm::ParameterSet & ps);
// class destructor
virtual ~L1TStage2BMTF();

// member functions
protected:
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override ;

// data members
private:  

  std::string monitorDir;
  edm::InputTag bmtfSource; 
  bool verbose ;
  edm::EDGetToken bmtfToken; 
  float global_phi;

  MonitorElement* bmtf_hwEta; 
  MonitorElement* bmtf_hwLocalPhi;
  MonitorElement* bmtf_hwGlobalPhi;
  MonitorElement* bmtf_hwPt;  
  MonitorElement* bmtf_hwQual;
  MonitorElement* bmtf_proc; 

  MonitorElement* bmtf_wedge_bx;
  MonitorElement* bmtf_hwEta_hwLocalPhi;
  MonitorElement* bmtf_hwPt_hwEta;
  MonitorElement* bmtf_hwPt_hwLocalPhi;

  MonitorElement* bmtf_hwEta_bx;  
  MonitorElement* bmtf_hwLocalPhi_bx;  
  MonitorElement* bmtf_hwPt_bx;   
  MonitorElement* bmtf_hwQual_bx; 

};

#endif
