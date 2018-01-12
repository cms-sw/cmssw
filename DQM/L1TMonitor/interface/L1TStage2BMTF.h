#ifndef DQM_L1TMonitor_L1TStage2BMTF_h
#define DQM_L1TMonitor_L1TStage2BMTF_h

/*
 * \file L1TStage2BMTF.h
 * \Author Esmaeel Eskandari Tadavani
 * \December 2015 
*/

// system requirements
#include <string>

// general requirements
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"

// stage2 requirements
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

// dqm requirements
#include "DQMServices/Core/interface/ConcurrentMonitorElement.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
//#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

namespace bmtfdqm {
  struct Histograms {
    ConcurrentMonitorElement bmtf_hwEta;
    ConcurrentMonitorElement bmtf_hwLocalPhi;
    ConcurrentMonitorElement bmtf_hwGlobalPhi;
    ConcurrentMonitorElement bmtf_hwPt;
    ConcurrentMonitorElement bmtf_hwQual;
    ConcurrentMonitorElement bmtf_proc;

    ConcurrentMonitorElement bmtf_wedge_bx;
    ConcurrentMonitorElement bmtf_hwEta_hwLocalPhi;
    ConcurrentMonitorElement bmtf_hwEta_hwGlobalPhi;

    ConcurrentMonitorElement bmtf_hwPt_hwEta;
    ConcurrentMonitorElement bmtf_hwPt_hwLocalPhi;

    ConcurrentMonitorElement bmtf_hwEta_bx;
    ConcurrentMonitorElement bmtf_hwLocalPhi_bx;
    ConcurrentMonitorElement bmtf_hwPt_bx;
    ConcurrentMonitorElement bmtf_hwQual_bx;

    //ConcurrentMonitorElement bmtf_twinmuxInput_PhiBX;
    //ConcurrentMonitorElement bmtf_twinmuxInput_PhiPhi;
    //ConcurrentMonitorElement bmtf_twinmuxInput_PhiPhiB;
    //ConcurrentMonitorElement bmtf_twinmuxInput_PhiQual;
    //ConcurrentMonitorElement bmtf_twinmuxInput_PhiStation;
    //ConcurrentMonitorElement bmtf_twinmuxInput_PhiSector;
    //ConcurrentMonitorElement bmtf_twinmuxInput_PhiWheel;
    //ConcurrentMonitorElement bmtf_twinmuxInput_PhiTrSeg;
    //ConcurrentMonitorElement bmtf_twinmuxInput_PhiWheel_PhiSector;

    //ConcurrentMonitorElement bmtf_twinmuxInput_TheBX;
    //ConcurrentMonitorElement bmtf_twinmuxInput_ThePhi;
    //ConcurrentMonitorElement bmtf_twinmuxInput_ThePhiB;
    //ConcurrentMonitorElement bmtf_twinmuxInput_TheQual;
    //ConcurrentMonitorElement bmtf_twinmuxInput_TheStation;
    //ConcurrentMonitorElement bmtf_twinmuxInput_TheSector;
    //ConcurrentMonitorElement bmtf_twinmuxInput_TheWheel;
    //ConcurrentMonitorElement bmtf_twinmuxInput_TheTrSeg;
    //ConcurrentMonitorElement bmtf_twinmuxInput_TheWheel_TheSector;
  };
}

// class declaration

class L1TStage2BMTF: public DQMGlobalEDAnalyzer<bmtfdqm::Histograms> {

public:

// class constructor
L1TStage2BMTF(const edm::ParameterSet & ps);
// class destructor
~L1TStage2BMTF() override;

// member functions
protected:
  void dqmAnalyze(const edm::Event&, const edm::EventSetup&, const bmtfdqm::Histograms&) const override;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&, bmtfdqm::Histograms&) const override;
  void bookHistograms(DQMStore::ConcurrentBooker&, const edm::Run&, const edm::EventSetup&, bmtfdqm::Histograms&) const override;

// data members
private:  

  std::string monitorDir;
  edm::InputTag bmtfSource; 
  //edm::InputTag bmtfSourceTwinMux1;
  //edm::InputTag bmtfSourceTwinMux2;
  bool verbose ;
  edm::EDGetToken bmtfToken;
  //edm::EDGetToken bmtfTokenTwinMux1;
  //edm::EDGetToken bmtfTokenTwinMux2;
};

#endif
