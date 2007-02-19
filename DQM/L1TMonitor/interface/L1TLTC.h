#ifndef L1TLTC_H
#define L1TLTC_H

/*
 * \file L1TLTC.h
 *
 * $Date: 2006/06/27 20:56:20 $
 * $Revision: 1.37 $
 * \author G. Della Ricca
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

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/LTCDigi/interface/LTCDigi.h"

//
// class decleration
//

class L1TLTC : public edm::EDAnalyzer {
public:
  explicit L1TLTC(const edm::ParameterSet&);
  ~L1TLTC();


  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob(void);
private:
  // ----------member data ---------------------------
  MonitorElement* h1;
  MonitorElement* h2;
  MonitorElement* h3;
  //MonitorElement* h4;
  MonitorElement* overlaps;
  MonitorElement* n_inhibit;
  MonitorElement* run;
  MonitorElement* event;
  MonitorElement* gps_time;
  float XMIN; float XMAX;
  // event counter
  int counter;
  // back-end interface
  DaqMonitorBEInterface * dbe;
  int nev_; // Number of events processed
  bool saveMe_; // save histograms or no?
  std::string rootFileName_;
};

#endif
