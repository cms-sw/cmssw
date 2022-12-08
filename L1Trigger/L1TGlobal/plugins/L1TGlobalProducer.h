#ifndef L1TGlobalProducer_h
#define L1TGlobalProducer_h

// Emulator for L1T Global
// author: Brian Winer  Ohio State

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "L1Trigger/L1TGlobal/interface/GlobalBoard.h"

#include "CondFormats/L1TObjects/interface/L1TGlobalParameters.h"
#include "L1Trigger/L1TGlobal/interface/GlobalParamsHelper.h"
#include "L1Trigger/L1TGlobal/interface/PrescalesVetosFractHelper.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1TGlobalParameters.h"
#include "CondFormats/DataRecord/interface/L1TGlobalParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1TGlobalPrescalesVetosFract.h"
#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosFractRcd.h"

class L1TGlobalParameters;
class L1GtParameters;
class L1GtBoardMaps;

class L1GtPrescaleFactors;
class L1GtTriggerMask;

class TriggerMenu;

// class declaration

class L1TGlobalProducer : public edm::stream::EDProducer<> {
public:
  explicit L1TGlobalProducer(const edm::ParameterSet&);
  ~L1TGlobalProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  /// cached stuff

  /// stable parameters
  const L1TGlobalParameters* m_l1GtStablePar;
  unsigned long long m_l1GtParCacheID;

  // trigger menu
  std::unique_ptr<TriggerMenu> m_l1GtMenu;
  unsigned long long m_l1GtMenuCacheID;

  // number of physics triggers
  unsigned int m_numberPhysTriggers;

  // number of DAQ partitions
  unsigned int m_numberDaqPartitions;

  // number of objects of each type
  int m_nrL1Mu;
  int m_nrL1MuShower;
  int m_nrL1EG;
  int m_nrL1Tau;

  int m_nrL1Jet;

  // ... the rest of the objects are global
  int m_ifMuEtaNumberBits;
  int m_ifCaloEtaNumberBits;

  ///    total number of Bx's in the event coming from EventSetup
  int m_totalBxInEvent;

  ///    active boards in L1 GT DAQ record
  uint16_t m_activeBoardsGtDaq;

  /// length of BST record (in bytes) from event setup
  unsigned int m_bstLengthBytes;

  /// board maps - cache only the record
  const L1GtBoardMaps* m_l1GtBM;
  unsigned long long m_l1GtBMCacheID;

  /// prescale factors
  const l1t::PrescalesVetosFractHelper* m_l1GtPrescalesVetosFract;
  unsigned long long m_l1GtPfAlgoCacheID;

  const std::vector<std::vector<double>>* m_prescaleFactorsAlgoTrig;
  std::vector<std::vector<double>> m_initialPrescaleFactorsAlgoTrig;

  uint m_currentLumi;

  /// trigger masks & veto masks
  const L1GtTriggerMask* m_l1GtTmAlgo;
  unsigned long long m_l1GtTmAlgoCacheID;

  const L1GtTriggerMask* m_l1GtTmVetoAlgo;
  unsigned long long m_l1GtTmVetoAlgoCacheID;

  const std::vector<unsigned int>* m_triggerMaskAlgoTrig;
  std::vector<unsigned int> m_initialTriggerMaskAlgoTrig;

  const std::vector<int>* m_triggerMaskVetoAlgoTrig;
  std::vector<int> m_initialTriggerMaskVetoAlgoTrig;

  std::unique_ptr<l1t::GlobalBoard> m_uGtBrd;

  /// input tag for muon collection from GMT
  edm::InputTag m_muInputTag;
  edm::InputTag m_muShowerInputTag;
  edm::EDGetTokenT<BXVector<l1t::Muon>> m_muInputToken;
  edm::EDGetTokenT<BXVector<l1t::MuonShower>> m_muShowerInputToken;

  /// input tag for calorimeter collections from GCT
  edm::InputTag m_egInputTag;
  edm::InputTag m_tauInputTag;
  edm::InputTag m_jetInputTag;
  edm::InputTag m_sumInputTag;
  edm::EDGetTokenT<BXVector<l1t::EGamma>> m_egInputToken;
  edm::EDGetTokenT<BXVector<l1t::Tau>> m_tauInputToken;
  edm::EDGetTokenT<BXVector<l1t::Jet>> m_jetInputToken;
  edm::EDGetTokenT<BXVector<l1t::EtSum>> m_sumInputToken;

  /// input tag for external conditions
  edm::InputTag m_extInputTag;
  edm::EDGetTokenT<BXVector<GlobalExtBlk>> m_extInputToken;

  /// logical flag to produce the L1 GT DAQ readout record
  bool m_produceL1GtDaqRecord;

  /// logical flag to produce the L1 GT object map record
  bool m_produceL1GtObjectMapRecord;

  /// logical flag to write the PSB content in the  L1 GT DAQ record
  bool m_writePsbL1GtDaqRecord;

  /// number of "bunch crossing in the event" (BxInEvent) to be emulated
  /// symmetric around L1Accept (BxInEvent = 0):
  ///    1 (BxInEvent = 0); 3 (F 0 1) (standard record); 5 (E F 0 1 2) (debug record)
  /// even numbers (except 0) "rounded" to the nearest lower odd number
  int m_emulateBxInEvent;

  /// Bx expected in Data coming to GT
  int m_L1DataBxInEvent;

  /// alternative for number of BX per active board in GT DAQ record: 0 or 1
  /// the position is identical with the active board bit
  unsigned int m_alternativeNrBxBoardDaq;

  /// length of BST record (in bytes) from parameter set
  int m_psBstLengthBytes;

  /// prescale set used
  unsigned int m_prescaleSet;

  /// run algorithm triggers
  ///     if true, unprescaled (all prescale factors 1)
  ///     will overwrite the event setup
  bool m_algorithmTriggersUnprescaled;

  ///     if true, unmasked - all enabled (all trigger masks set to 0)
  ///     will overwrite the event setup
  bool m_algorithmTriggersUnmasked;

  /// verbosity level
  int m_verbosity;
  bool m_printL1Menu;
  bool m_isDebugEnabled;

  bool m_getPrescaleColumnFromData;
  bool m_requireMenuToMatchAlgoBlkInput;
  edm::InputTag m_algoblkInputTag;
  edm::EDGetToken m_algoblkInputToken;

  edm::ESGetToken<L1TGlobalParameters, L1TGlobalParametersRcd> m_l1GtStableParToken;
  edm::ESGetToken<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd> m_l1GtMenuToken;
  edm::ESGetToken<L1TGlobalPrescalesVetosFract, L1TGlobalPrescalesVetosFractRcd> m_l1GtPrescaleVetosToken;

  // disables resetting the prescale counters each lumisection (needed for offline)
  bool m_resetPSCountersEachLumiSec;

  // initialise prescale counters with a semi-random value in the range [0, prescale*10^precision - 1];
  // if false, the prescale counters are initialised to zero
  bool m_semiRandomInitialPSCounters;

  // switch to load muon showers in the global board
  bool m_useMuonShowers;
};

#endif  // L1TGlobalProducer_h
