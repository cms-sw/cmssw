#ifndef L1TDTTPG_H
#define L1TDTTPG_H

/*
 * \file L1TDTTPG.h
 *
 * $Date: 2007/02/22 19:43:52 $
 * $Revision: 1.4 $
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



#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

//
// class decleration
//

class L1TDTTPG : public edm::EDAnalyzer {

public:

// Constructor
L1TDTTPG(const edm::ParameterSet& ps);

// Destructor
virtual ~L1TDTTPG();

protected:
// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

private:
  // ----------member data ---------------------------
  DaqMonitorBEInterface * dbe;

  MonitorElement* dttpgphbx;  
  MonitorElement* dttpgphwheel;  
  MonitorElement* dttpgphsector;  
  MonitorElement* dttpgphstation;  
  MonitorElement* dttpgphphi;  
  MonitorElement* dttpgphphiB;  
  MonitorElement* dttpgphquality;  
  MonitorElement* dttpgphts2tag;  
  MonitorElement* dttpgphbxcnt;  
  MonitorElement* dttpgphntrack;  

  MonitorElement* dttpgthbx;  
  MonitorElement* dttpgthwheel;  
  MonitorElement* dttpgthsector;  
  MonitorElement* dttpgthstation;  
  MonitorElement* dttpgththeta;  
  MonitorElement* dttpgthquality;    
  MonitorElement* dttpgthntrack;  

  MonitorElement *dttf_p_phi;
  MonitorElement *dttf_p_pt ;
  MonitorElement *dttf_p_q;
  MonitorElement *dttf_p_qual;

  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;
  edm::InputTag dttpgSource_;
};

#endif
