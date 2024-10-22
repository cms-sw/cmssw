/**
 * \author L. Gray
 * \class CSCTFSectorProcessor.h
 *
 * A class that represents a sector processor board.
 */

#ifndef CSCTrackFinder_CSCTFSectorProcessor_h
#define CSCTrackFinder_CSCTFSectorProcessor_h

#include <vector>
#include <map>
#include <string>
#include "CondFormats/L1TObjects/interface/L1MuCSCTFConfiguration.h"
#include "CondFormats/DataRecord/interface/L1MuCSCTFConfigurationRcd.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTFSPCoreLogic.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h"
///KK
#include "FWCore/Framework/interface/EventSetup.h"
///

class CSCTFSectorProcessor {
public:
  struct Tokens {
    CSCTFPtLUT::Tokens ptLUT;
    edm::ESGetToken<L1MuCSCTFConfiguration, L1MuCSCTFConfigurationRcd> config;
  };

  static Tokens consumes(const edm::ParameterSet& pset, edm::ConsumesCollector iC);

  CSCTFSectorProcessor(const unsigned& endcap,
                       const unsigned& sector,
                       const edm::ParameterSet& pset,
                       bool tmb07,
                       const L1MuTriggerScales* scales,
                       const L1MuTriggerPtScale* ptScale);

  ///KK
  void initialize(const edm::EventSetup& c, const Tokens& tokens);
  ///

  ~CSCTFSectorProcessor();

  //returns 0 for normal fail, 1 for success, and -1 for exception
  // on -1, Producer should produce empty collections for event
  int run(const CSCTriggerContainer<csctf::TrackStub>&);

  CSCTriggerContainer<csc::L1Track> tracks() const { return l1_tracks; }
  std::vector<csctf::TrackStub> filteredStubs() const { return stub_vec_filtered; }

  CSCTriggerContainer<csctf::TrackStub> dtStubs() const { return dt_stubs; }

  int minBX() const { return m_minBX; }
  int maxBX() const { return m_maxBX; }

  void readParameters(const edm::ParameterSet& pset);

  void printDisclaimer(int firmSP, int firmFA);

private:
  // disallow copy and assignment
  CSCTFSectorProcessor& operator=(const CSCTFSectorProcessor& rhs) { return *this; };
  CSCTFSectorProcessor(const CSCTFSectorProcessor& par) {}

  bool m_gangedME1a;

  bool initializeFromPSet;
  unsigned m_endcap, m_sector, TMB07;
  unsigned m_latency;

  // All parameters below are signed to allow for uninitialized (<0) state
  int m_bxa_depth, m_allowALCTonly, m_allowCLCTonly, m_preTrigger;
  int m_minBX, m_maxBX;
  int m_etawin[7], m_etamin[8], m_etamax[8];
  int m_mindphip, m_mindetap;
  int m_mindeta12_accp, m_maxdeta12_accp, m_maxdphi12_accp;
  int m_mindeta13_accp, m_maxdeta13_accp, m_maxdphi13_accp;
  int m_mindeta112_accp, m_maxdeta112_accp, m_maxdphi112_accp;
  int m_mindeta113_accp, m_maxdeta113_accp, m_maxdphi113_accp;
  int m_mindphip_halo, m_mindetap_halo;
  int m_straightp, m_curvedp;
  int m_mbaPhiOff, m_mbbPhiOff;
  int m_widePhi;

  //  following parameters were moved here from the CSCTFTrackBuilder because they naturally belong here
  int QualityEnableME1a, QualityEnableME1b, QualityEnableME1c, QualityEnableME1d, QualityEnableME1e, QualityEnableME1f;
  int QualityEnableME2a, QualityEnableME2b, QualityEnableME2c;
  int QualityEnableME3a, QualityEnableME3b, QualityEnableME3c;
  int QualityEnableME4a, QualityEnableME4b, QualityEnableME4c;
  int kill_fiber;
  int run_core;
  int trigger_on_ME1a, trigger_on_ME1b, trigger_on_ME2, trigger_on_ME3, trigger_on_ME4;
  int trigger_on_MB1a, trigger_on_MB1d;
  unsigned int singlesTrackOutput;
  int rescaleSinglesPhi;

  int m_firmSP, m_firmFA, m_firmDD, m_firmVM;

  CSCTriggerContainer<csc::L1Track> l1_tracks;     // fully defined L1Tracks
  CSCTriggerContainer<csctf::TrackStub> dt_stubs;  // Track Stubs to be sent to the DTTF
  std::vector<csctf::TrackStub>
      stub_vec_filtered;  // Collectin of stubs after applying kill_fiber and QualityEnable masks

  static const std::string FPGAs[5];

  std::map<std::string, CSCSectorReceiverLUT*> srLUTs_;  // indexed by FPGA
  CSCTFSPCoreLogic* core_;
  CSCTFPtLUT* ptLUT_;

  // firmware map
  std::map<int, int> firmSP_Map;
  bool isCoreVerbose;
  bool initFail_;
};

#endif
