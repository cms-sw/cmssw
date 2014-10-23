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

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
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
 
#include <iostream>
#include <fstream>
#include <vector>

//
// class decleration
//

class L1TCSCTF : public edm::EDAnalyzer {

 public:

  // Constructor
  L1TCSCTF(const edm::ParameterSet& ps);

  // Destructor
  virtual ~L1TCSCTF();

 protected:
  // Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  // BeginJob
  void beginJob(void);

  // EndJob
  void endJob(void);

 private:
  // ----------member data ---------------------------
  DQMStore * dbe;

  MonitorElement* csctfntrack;
  MonitorElement* csctfbx;
  MonitorElement* csctfbx_H;

  MonitorElement* csctferrors;
  MonitorElement* csctfoccupancies;
  MonitorElement* csctfoccupancies_H;
  
  //MonitorElement* haloDelEta112;
  //MonitorElement* haloDelEta12;
  //MonitorElement* haloDelEta113;
  //MonitorElement* haloDelEta13;
  
  MonitorElement* csctfChamberOccupancies;
  MonitorElement* csctfTrackPhi; //all tracks but halo
  MonitorElement* csctfTrackEta; //all tracks but halo
  MonitorElement* csctfTrackEtaLowQ;  //all tracks but halo
  MonitorElement* csctfTrackEtaHighQ; //all tracks but halo
  MonitorElement* csctfTrackPhi_H; //halo tracks only
  MonitorElement* csctfTrackEta_H; //halo tracks only
  MonitorElement* cscTrackStubNumbers;
  MonitorElement* csctfTrackM;
  MonitorElement* trackModeVsQ;
  MonitorElement* csctfAFerror;

  // NEW: CSC EVENT LCT PLOTS, Renjie Wang
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



  // 1-> 6 plus endcap
  // 7->12 minus endcap
  MonitorElement* DTstubsTimeTrackMenTimeArrival[12];
  int BxInEvent_; //bx of the CSC muon candidate
  bool isCSCcand_;//does GMT readout window have a CSC cand?

  int L1ABXN;

  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  std::ofstream logFile_;
  edm::InputTag gmtProducer, lctProducer, trackProducer, statusProducer, mbProducer;
  bool gangedME11a_;

  CSCSectorReceiverLUT *srLUTs_[5];

  const L1MuTriggerScales  *ts;
  const L1MuTriggerPtScale *tpts;
  unsigned long long m_scalesCacheID ;
  unsigned long long m_ptScaleCacheID ;

  //define Token(-s)
  edm::EDGetTokenT<L1MuGMTReadoutCollection> gmtProducerToken_;
  edm::EDGetTokenT<L1CSCStatusDigiCollection> statusToken_;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> corrlctsToken_;
  edm::EDGetTokenT<L1CSCTrackCollection> tracksToken_;
  edm::EDGetTokenT<CSCTriggerContainer<csctf::TrackStub> > dtStubsToken_;
  edm::EDGetTokenT<L1CSCTrackCollection> mbtracksToken_;
};

#endif
