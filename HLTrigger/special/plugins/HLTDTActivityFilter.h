#ifndef HLTDTActivityFilter_h
#define HLTDTActivityFilter_h
// -*- C++ -*-
//
// Package:    HLTDTActivityFilter
// Class:      HLTDTActivityFilter
//

/*

Description: Filter to select events with activity in the muon barrel system

*/

//
// Original Author:  Carlo Battilana
//         Created:  Tue Jan 22 13:55:00 CET 2008
//
//

// Fwk header files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/DTDigi/interface/DTLocalTriggerCollection.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"

// c++ header files
#include <bitset>
#include <string>

class DTGeometry;
class L1MuRegionalCand;

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTDTActivityFilter : public HLTFilter {
public:
  explicit HLTDTActivityFilter(const edm::ParameterSet &);
  ~HLTDTActivityFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  bool hltFilter(edm::Event &,
                 const edm::EventSetup &,
                 trigger::TriggerFilterObjectWithRefs &filterproduct) const override;

  bool hasActivity(const std::bitset<4> &) const;
  bool matchChamber(uint32_t rawId, L1MuRegionalCand const &rpcTrig, DTGeometry const *dtGeom) const;

  enum activityType { DCC = 0, DDU = 1, RPC = 2, DIGI = 3 };

  // ----------member data ---------------------------

  edm::InputTag inputTag_[4];
  bool process_[4];
  std::bitset<15> activeSecs_;

  edm::EDGetTokenT<L1MuDTChambPhContainer> inputDCCToken_;
  edm::EDGetTokenT<DTLocalTriggerCollection> inputDDUToken_;
  edm::EDGetTokenT<L1MuGMTReadoutCollection> inputRPCToken_;
  edm::EDGetTokenT<DTDigiCollection> inputDigiToken_;

  bool orTPG_;
  bool orRPC_;
  bool orDigi_;

  int minQual_;
  int maxStation_;
  int minBX_[3];
  int maxBX_[3];
  int minActiveChambs_;
  int minChambLayers_;

  float maxDeltaPhi_;
  float maxDeltaEta_;
};

#endif
