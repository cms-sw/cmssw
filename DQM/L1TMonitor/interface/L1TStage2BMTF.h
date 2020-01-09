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
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

// dqm requirements
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "L1Trigger/L1TMuon/interface/MuonRawDigiTranslator.h"
#include "L1Trigger/L1TMuon/interface/RegionalMuonRawDigiTranslator.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

// class decleration

class L1TStage2BMTF : public DQMEDAnalyzer {
public:
  // class constructor
  L1TStage2BMTF(const edm::ParameterSet& ps);
  // class destructor
  ~L1TStage2BMTF() override;

  // member functions
protected:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;

  // data members
private:
  std::string monitorDir;
  edm::InputTag bmtfSource;
  //  edm::InputTag bmtfSourceTwinMux1;
  //  edm::InputTag bmtfSourceTwinMux2;
  bool verbose;
  bool kalman;
  edm::EDGetToken bmtfToken;
  // edm::EDGetToken bmtfTokenTwinMux1;
  //  edm::EDGetToken bmtfTokenTwinMux2;
  float global_phi;

  MonitorElement* bmtf_hwEta;
  MonitorElement* bmtf_hwLocalPhi;
  MonitorElement* bmtf_hwGlobalPhi;
  MonitorElement* bmtf_hwPt;
  MonitorElement* bmtf_hwQual;
  MonitorElement* bmtf_proc;

  MonitorElement* bmtf_wedge_bx;
  MonitorElement* bmtf_hwEta_hwLocalPhi;
  MonitorElement* bmtf_hwEta_hwGlobalPhi;

  MonitorElement* bmtf_hwPt_hwEta;
  MonitorElement* bmtf_hwPt_hwLocalPhi;

  MonitorElement* bmtf_hwEta_bx;
  MonitorElement* bmtf_hwLocalPhi_bx;
  MonitorElement* bmtf_hwPt_bx;
  MonitorElement* bmtf_hwQual_bx;

  MonitorElement* bmtf_hwDXY;
  MonitorElement* bmtf_hwPt2;

  /* MonitorElement* bmtf_twinmuxInput_PhiBX; */
  /* MonitorElement* bmtf_twinmuxInput_PhiPhi; */
  /* MonitorElement* bmtf_twinmuxInput_PhiPhiB; */
  /* MonitorElement* bmtf_twinmuxInput_PhiQual; */
  /* MonitorElement* bmtf_twinmuxInput_PhiStation; */
  /* MonitorElement* bmtf_twinmuxInput_PhiSector; */
  /* MonitorElement* bmtf_twinmuxInput_PhiWheel; */
  /* MonitorElement* bmtf_twinmuxInput_PhiTrSeg; */
  /* MonitorElement* bmtf_twinmuxInput_PhiWheel_PhiSector; */

  /* MonitorElement* bmtf_twinmuxInput_TheBX; */
  /* MonitorElement* bmtf_twinmuxInput_ThePhi; */
  /* MonitorElement* bmtf_twinmuxInput_ThePhiB; */
  /* MonitorElement* bmtf_twinmuxInput_TheQual; */
  /* MonitorElement* bmtf_twinmuxInput_TheStation; */
  /* MonitorElement* bmtf_twinmuxInput_TheSector; */
  /* MonitorElement* bmtf_twinmuxInput_TheWheel; */
  /* MonitorElement* bmtf_twinmuxInput_TheTrSeg; */
  /* MonitorElement* bmtf_twinmuxInput_TheWheel_TheSector; */
};

#endif
