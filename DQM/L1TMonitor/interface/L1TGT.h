#ifndef L1TGT_H
#define L1TGT_H

/**
 * \class L1TGT
 *
 *
 * Description: DQM for L1 Global Trigger.
 *
 * \author J. Berryhill, I. Mikulec
 * \author Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// system include files
#include <memory>
#include <unistd.h>
#include <vector>
#include <utility>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

//L1 trigger includes
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

//
// class declaration
//

class L1TGT : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  // constructor
  L1TGT(const edm::ParameterSet& ps);

  // destructor
  ~L1TGT() override;

protected:
  //virtual void beginJob();
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  /// end section
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) final {}
  void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;

private:
  /// book all histograms for the module

  bool isActive(int word, int bit);
  // Active boards DAQ record bit number:
  // 0 FDL
  // 1 PSB_0 9 Techn.Triggers for FDL
  // 2 PSB_1 13 Calo data for GTL
  // 3 PSB_2 14 Calo data for GTL
  // 4 PSB_3 15 Calo data for GTL
  // 5 PSB_4 19 M/Q bits for GMT
  // 6 PSB_5 20 M/Q bits for GMT
  // 7 PSB_6 21 M/Q bits for GMT
  // 8 GMT
  enum activeDAQ { FDL = 0, PSB9, PSB13, PSB14, PSB15, PSB19, PSB20, PSB21, GMT };
  // Active boards EVM record bit number:
  // 0 TCS
  // 1 FDL
  enum activeEVM { TCS, FDLEVM };

  // count the number of indices per Ls for prescale factor sets
  // if no errors, it must be 1
  void countPfsIndicesPerLs();

private:
  /// input parameters

  /// input tag for L1 GT DAQ readout record
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> gtSource_L1GT_;
  edm::EDGetTokenT<L1MuGMTReadoutCollection> gtSource_L1MuGMT_;

  /// input tag for L1 GT EVM readout record
  edm::EDGetTokenT<L1GlobalTriggerEvmReadoutRecord> gtEvmSource_;

  /// switches to choose the running of various methods
  bool m_runInEventLoop;
  bool m_runInEndLumi;

  /// verbosity switch
  bool verbose_;

private:
  MonitorElement* algo_bits;
  MonitorElement* algo_bits_corr;
  MonitorElement* tt_bits;
  MonitorElement* tt_bits_corr;
  MonitorElement* algo_tt_bits_corr;
  MonitorElement* algo_bits_lumi;
  MonitorElement* tt_bits_lumi;
  MonitorElement* event_type;

  MonitorElement* event_number;
  MonitorElement* event_lumi;
  MonitorElement* trigger_number;
  MonitorElement* trigger_lumi;
  MonitorElement* evnum_trignum_lumi;
  MonitorElement* orbit_lumi;
  MonitorElement* setupversion_lumi;

  MonitorElement* gtfe_bx;
  MonitorElement* dbx_module;

  MonitorElement* BST_MasterStatus;
  MonitorElement* BST_turnCountNumber;
  MonitorElement* BST_lhcFillNumber;
  MonitorElement* BST_beamMode;
  MonitorElement* BST_beamMomentum;
  MonitorElement* BST_intensityBeam1;
  MonitorElement* BST_intensityBeam2;
  MonitorElement* gpsfreq;
  MonitorElement* gpsfreqwide;
  MonitorElement* gpsfreqlum;

  MonitorElement* m_monL1PrescaleFactorSet;
  MonitorElement* m_monL1PfIndicesPerLs;

  MonitorElement* m_monOrbitNrDiffTcsFdlEvm;
  MonitorElement* m_monLsNrDiffTcsFdlEvm;
  // maximum difference in orbit number, luminosity number
  // histogram range: -(MaxOrbitNrDiffTcsFdlEvm+1), (MaxOrbitNrDiffTcsFdlEvm+1)
  //   if value is greater than the maximum difference, fill an entry in the last but one bin
  //   if value is smaller than the negative value of maximum difference, fill an entry
  //     in the second bin
  //   if no value can be retrieved for TCS, fill an entry in the first bin
  //   if no value can be retrieved for FDL, fill an entry in the last bin
  static const int MaxOrbitNrDiffTcsFdlEvm;
  static const int MaxLsNrDiffTcsFdlEvm;

  MonitorElement* m_monOrbitNrDiffTcsFdlEvmLs;
  MonitorElement* m_monLsNrDiffTcsFdlEvmLs;

  MonitorElement* h_L1AlgoBX1;
  MonitorElement* h_L1AlgoBX2;
  MonitorElement* h_L1AlgoBX3;
  MonitorElement* h_L1AlgoBX4;
  MonitorElement* h_L1TechBX;

  //MonitorElement* m_monDiffEvmDaqFdl;

private:
  /// number of events processed
  int m_nrEvJob;
  int m_nrEvRun;

  /// histogram folder for L1 GT plots
  std::string m_histFolder;

  uint64_t preGps_;
  uint64_t preOrb_;

  std::string algoBitToName[128];
  std::string techBitToName[64];
  std::map<std::string, bool> l1TriggerDecision, l1TechTriggerDecision;
  std::map<std::string, bool>::iterator trig_iter;

  std::vector<std::pair<int, int> > m_pairLsNumberPfIndex;
  typedef std::vector<std::pair<int, int> >::const_iterator CItVecPair;
};

#endif
