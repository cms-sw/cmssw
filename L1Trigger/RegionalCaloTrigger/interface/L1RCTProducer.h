#ifndef L1RCTProducer_h
#define L1RCTProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

// default scales
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"

#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"
#include "CondFormats/DataRecord/interface/L1RCTNoisyChannelMaskRcd.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"
#include "CondFormats/L1TObjects/interface/L1RCTNoisyChannelMask.h"
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"

#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"

#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>
#include <memory>

class L1RCT;
class L1RCTLookupTables;

class L1RCTProducer : public edm::stream::EDProducer<> {
public:
  explicit L1RCTProducer(const edm::ParameterSet &ps);

  void beginRun(edm::Run const &r, const edm::EventSetup &c) final;
  void beginLuminosityBlock(edm::LuminosityBlock const &lumiSeg, const edm::EventSetup &context) final;
  void produce(edm::Event &e, const edm::EventSetup &c) final;

  void updateConfiguration(const edm::EventSetup &);

  void updateFedVector(const edm::EventSetup &, bool getFromOmds, int);
  const std::vector<int> getFedVectorFromRunInfo(const edm::EventSetup &);
  const std::vector<int> getFedVectorFromOmds(const edm::EventSetup &);

  void printFedVector(const std::vector<int> &);
  void printUpdatedFedMask();
  void printUpdatedFedMaskVerbose();

private:
  std::unique_ptr<L1RCTLookupTables> rctLookupTables;
  std::unique_ptr<L1RCT> rct;
  bool useEcal;
  bool useHcal;
  std::vector<edm::InputTag> ecalDigis;
  std::vector<edm::InputTag> hcalDigis;
  std::vector<int> bunchCrossings;
  bool getFedsFromOmds;
  unsigned int queryDelayInLS;
  unsigned int queryIntervalInLS;
  std::string conditionsLabel;

  // Create a channel mask object to be updated at every Run....
  std::unique_ptr<L1RCTChannelMask> fedUpdatedMask;

  enum crateSection { c_min, ebOddFed = c_min, ebEvenFed, eeFed, hbheFed, hfFed, hfFedUp, c_max = hfFedUp };

  static const int crateFED[18][6];

  static const int minBarrel = 1;
  static const int maxBarrel = 17;
  static const int minEndcap = 17;
  static const int maxEndcap = 28;
  static const int minHF = 29;
  static const int maxHF = 32;
};

#endif
