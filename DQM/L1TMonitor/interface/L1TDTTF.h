#ifndef L1TDTTF_H
#define L1TDTTF_H

/*
 * \file L1TDTTF.h
 *
 * $Date: 2008/03/10 09:30:27 $
 * $Revision: 1.8 $
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

//
// class decleration
//

class L1TDTTF : public edm::EDAnalyzer {

 public:

  // Constructor
  L1TDTTF(const edm::ParameterSet& ps);

  // Destructor
  virtual ~L1TDTTF();

 protected:
  // Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  // BeginJob
  void beginJob(const edm::EventSetup& c);

  // EndJob
  void endJob(void);

 private:

  void setMapPhLabel(MonitorElement *me);
  void setMapThLabel(MonitorElement *me);

  // ----------member data ---------------------------
  DQMStore * dbe;
  std::string l1tinfofolder;
  std::string l1tsubsystemfolder;
  
  MonitorElement* dttpgphbx[8];  
  MonitorElement* dttpgphbxcomp;
  MonitorElement* dttpgphntrack;
  MonitorElement* dttpgthntrack;  

  MonitorElement* dttpgphwheel[3];
  MonitorElement* dttpgphsector[3][5];
  MonitorElement* dttpgphstation[3][5][12];
  MonitorElement* dttpgphsg1phiAngle[3][5][12][5];
  MonitorElement* dttpgphsg1phiBandingAngle[3][5][12][5];
  MonitorElement* dttpgphsg1quality[3][5][12][5];
  MonitorElement* dttpgphsg2phiAngle[3][5][12][5];
  MonitorElement* dttpgphsg2phiBandingAngle[3][5][12][5];
  MonitorElement* dttpgphsg2quality[3][5][12][5];
  MonitorElement* dttpgphts2tag[3][5][12][5];
  MonitorElement* dttpgphmapbx[3];
  MonitorElement* bxnumber[5][12][5];

  MonitorElement* dttpgthbx[3];  
  MonitorElement* dttpgthwheel[3];  
  MonitorElement* dttpgthsector[3][6];  
  MonitorElement* dttpgthstation[3][6][12];  
  MonitorElement* dttpgththeta[3][6][12][4];  
  MonitorElement* dttpgthquality[3][6][12][4];   
  MonitorElement* dttpgthmap;
  MonitorElement* dttpgthmapbx[3];

  MonitorElement* dttf_p_phi[3][6][12];
  MonitorElement* dttf_p_qual[3][6][12];
  MonitorElement* dttf_p_q[3][6][12];
  MonitorElement* dttf_p_pt[3][6][12];
  

  MonitorElement* dttpgphmap;
  MonitorElement* dttpgphmap2nd;
  MonitorElement* dttpgphmapcorr;
  MonitorElement* dttpgphbestmap;
  MonitorElement* dttpgphbestmapcorr;


  MonitorElement* dttpgthmaph;
  MonitorElement* dttpgthbestmap;
  MonitorElement* dttpgthbestmaph;

  
  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;
  edm::InputTag dttpgSource_;
};

#endif
