#ifndef L1TCSCTF_H
#define L1TCSCTF_H

/*
 * \file L1TCSCTF.h
 *
 * $Date: 2009/11/19 14:30:24 $
 * $Revision: 1.14 $
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

// KK_start: Sector Receiver LUT class to transform wire/strip numbers to eta/phi observables
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
// KK_end

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

  // MonitorElement* csctfetavalue[3];
  // MonitorElement* csctfphivalue[3];
  // MonitorElement* csctfptvalue[3];
  // MonitorElement* csctfchargevalue[3];
  // MonitorElement* csctfquality[3];
  MonitorElement* csctfntrack;
  MonitorElement* csctfbx;

  // KK_start: see source for description
  MonitorElement* csctferrors;
  MonitorElement* csctfoccupancies;
  CSCSectorReceiverLUT *srLUTs_[5];
  // KK_end
  
  // JAG
  MonitorElement* haloDelEta23;
  MonitorElement* csctfChamberOccupancies;
  MonitorElement* csctfTrackPhi;
  MonitorElement* csctfTrackEta;
  MonitorElement* cscTrackStubNumbers;
  MonitorElement* csctfTrackQ;
  MonitorElement* csctfAFerror;
  // JAG

  // GP
  // 1-> 6 plus endcap
  // 7->12 minus endcap
  MonitorElement* DTstubsTimeTrackMenTimeArrival[12];
  int BxInEvent_; //bx of the CSC muon candidate
  bool isCSCcand_;//does GMT readout window have a CSC cand?

  MonitorElement* csctfHaloL1ABXN;
  MonitorElement* csctfCoincL1ABXN;
  int L1ABXN;
  // GP_end

  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;
  edm::InputTag gmtProducer, lctProducer, trackProducer, statusProducer, mbProducer;
};

#endif
