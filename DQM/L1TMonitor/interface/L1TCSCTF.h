#ifndef L1TCSCTF_H
#define L1TCSCTF_H

/*
 * \file L1TCSCTF.h
 *
 * \author J. Berryhill
 *
*/

// system include files
#include <memory>
#include <unistd.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

// Sector Receiver LUT class to transform wire/strip numbers to eta/phi observables
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"

#include "DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include <iostream>
#include <fstream>
#include <vector>

//
// class decleration
//

class L1TCSCTF : public DQMEDAnalyzer {
public:
  // Constructor
  L1TCSCTF(const edm::ParameterSet& ps);

  // Destructor
  ~L1TCSCTF() override;

protected:
  // Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) override;

private:
  // ----------member data ---------------------------

  MonitorElement* csctfntrack;
  MonitorElement* csctfbx;
  MonitorElement* csctfbx_H;

  MonitorElement* csctferrors;
  MonitorElement* csctfoccupancies;
  MonitorElement* csctfoccupancies_H;

  MonitorElement* csctferrors_mpc;
  MonitorElement* cscWireStripOverflow;

  //MonitorElement* runId_;
  //MonitorElement* lumisecId_;

  //MonitorElement* haloDelEta112;
  //MonitorElement* haloDelEta12;
  //MonitorElement* haloDelEta113;
  //MonitorElement* haloDelEta13;

  MonitorElement* csctfChamberOccupancies;
  MonitorElement* csctfTrackPhi;       //all tracks but halo
  MonitorElement* csctfTrackEta;       //all tracks but halo
  MonitorElement* csctfTrackEtaLowQ;   //all tracks but halo
  MonitorElement* csctfTrackEtaHighQ;  //all tracks but halo
  MonitorElement* csctfTrackPhi_H;     //halo tracks only
  MonitorElement* csctfTrackEta_H;     //halo tracks only
  MonitorElement* cscTrackStubNumbers;
  MonitorElement* csctfTrackM;
  MonitorElement* trackModeVsQ;
  MonitorElement* csctfAFerror;

  // NEW: CSC EVENT LCT PLOTS
  MonitorElement* csctflcts;

  // PLOTS SPECIFICALLY FOR ME1/1
  MonitorElement* me11_lctStrip;
  MonitorElement* me11_lctWire;
  MonitorElement* me11_lctLocalPhi;
  MonitorElement* me11_lctPackedPhi;
  MonitorElement* me11_lctGblPhi;
  MonitorElement* me11_lctGblEta;

  // PLOTS SPECIFICALLY FOR ME4/2
  MonitorElement* me42_lctGblPhi;
  MonitorElement* me42_lctGblEta;

  // WG AND STRIP PLOTS FOR ALL CHAMBERS
  MonitorElement* csc_strip_MEplus11;
  MonitorElement* csc_strip_MEplus12;
  MonitorElement* csc_strip_MEplus13;
  MonitorElement* csc_strip_MEplus21;
  MonitorElement* csc_strip_MEplus22;
  MonitorElement* csc_strip_MEplus31;
  MonitorElement* csc_strip_MEplus32;
  MonitorElement* csc_strip_MEplus41;
  MonitorElement* csc_strip_MEplus42;

  MonitorElement* csc_strip_MEminus11;
  MonitorElement* csc_strip_MEminus12;
  MonitorElement* csc_strip_MEminus13;
  MonitorElement* csc_strip_MEminus21;
  MonitorElement* csc_strip_MEminus22;
  MonitorElement* csc_strip_MEminus31;
  MonitorElement* csc_strip_MEminus32;
  MonitorElement* csc_strip_MEminus41;
  MonitorElement* csc_strip_MEminus42;

  MonitorElement* csc_wire_MEplus11;
  MonitorElement* csc_wire_MEplus12;
  MonitorElement* csc_wire_MEplus13;
  MonitorElement* csc_wire_MEplus21;
  MonitorElement* csc_wire_MEplus22;
  MonitorElement* csc_wire_MEplus31;
  MonitorElement* csc_wire_MEplus32;
  MonitorElement* csc_wire_MEplus41;
  MonitorElement* csc_wire_MEplus42;

  MonitorElement* csc_wire_MEminus11;
  MonitorElement* csc_wire_MEminus12;
  MonitorElement* csc_wire_MEminus13;
  MonitorElement* csc_wire_MEminus21;
  MonitorElement* csc_wire_MEminus22;
  MonitorElement* csc_wire_MEminus31;
  MonitorElement* csc_wire_MEminus32;
  MonitorElement* csc_wire_MEminus41;
  MonitorElement* csc_wire_MEminus42;

  // 1-> 6 plus endcap
  // 7->12 minus endcap
  MonitorElement* DTstubsTimeTrackMenTimeArrival[12];
  int BxInEvent_;   //bx of the CSC muon candidate
  bool isCSCcand_;  //does GMT readout window have a CSC cand?

  int L1ABXN;

  int nev_;                 // Number of events processed
  std::string outputFile_;  //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  std::ofstream logFile_;
  edm::InputTag gmtProducer, lctProducer, trackProducer, statusProducer, mbProducer;
  bool gangedME11a_;  // needed this be set false for Run2

  CSCSectorReceiverLUT* srLUTs_[5][2][6];

  const L1MuTriggerScales* ts;
  const L1MuTriggerPtScale* tpts;
  unsigned long long m_scalesCacheID;
  unsigned long long m_ptScaleCacheID;

  //define Token(-s)
  edm::EDGetTokenT<L1MuGMTReadoutCollection> gmtProducerToken_;
  edm::EDGetTokenT<L1CSCStatusDigiCollection> statusToken_;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> corrlctsToken_;
  edm::EDGetTokenT<L1CSCTrackCollection> tracksToken_;
  edm::EDGetTokenT<CSCTriggerContainer<csctf::TrackStub> > dtStubsToken_;
  edm::EDGetTokenT<L1CSCTrackCollection> mbtracksToken_;
  edm::ESGetToken<L1MuTriggerScales, L1MuTriggerScalesRcd> l1muTscalesToken_;
  edm::ESGetToken<L1MuTriggerPtScale, L1MuTriggerPtScaleRcd> ptscalesToken_;
};

#endif
