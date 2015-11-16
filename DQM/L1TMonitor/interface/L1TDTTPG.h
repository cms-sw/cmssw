#ifndef L1TDTTPG_H
#define L1TDTTPG_H

/*
 * \file L1TDTTPG.h
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

// L1 containers
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

//
// class decleration
//

class L1TDTTPG : public DQMEDAnalyzer {

 public:

  // Constructor
  L1TDTTPG(const edm::ParameterSet& ps);

  // Destructor
  virtual ~L1TDTTPG();

 protected:
  // Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  
  // BeginRun
  virtual void bookHistograms(DQMStore::IBooker &ibooker, edm::Run const&, edm::EventSetup const&) override;
  virtual void dqmBeginRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);


 private:

  void setMapPhLabel(MonitorElement *me);
  void setMapThLabel(MonitorElement *me);

  // ----------member data ---------------------------

  MonitorElement* dttpgphbx[8];  
  MonitorElement* dttpgphbxcomp;
  MonitorElement* dttpgphwheel[3];  
  MonitorElement* dttpgphsector[3];  
  MonitorElement* dttpgphstation[3];  
  /*   MonitorElement* dttpgphphi[3];   */
  /*   MonitorElement* dttpgphphiB[3];   */
  MonitorElement* dttpgphquality[3];  
  MonitorElement* dttpgphts2tag[3];  
  /*   MonitorElement* dttpgphbxcnt[3];   */
  MonitorElement* dttpgphntrack;
  MonitorElement* dttpgphmap;
  MonitorElement* dttpgphmapbx[3];
  MonitorElement* dttpgphmap2nd;
  MonitorElement* dttpgphmapcorr;
  MonitorElement* dttpgphbestmap;
  MonitorElement* dttpgphbestmapcorr;


  MonitorElement* dttpgthbx[3];  
  MonitorElement* dttpgthwheel[3];  
  MonitorElement* dttpgthsector[3];  
  MonitorElement* dttpgthstation[3];  
  MonitorElement* dttpgththeta[3];  
  MonitorElement* dttpgthquality[3];    
  MonitorElement* dttpgthntrack;  
  MonitorElement* dttpgthmap;
  MonitorElement* dttpgthmapbx[3];
  MonitorElement* dttpgthmaph;
  MonitorElement* dttpgthbestmap;
  MonitorElement* dttpgthbestmaph;

  MonitorElement *dttf_p_phi[3];
  MonitorElement *dttf_p_pt[3];
  MonitorElement *dttf_p_q[3];
  MonitorElement *dttf_p_qual[3];

  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  std::ofstream logFile_;
  edm::EDGetTokenT<L1MuDTChambPhContainer> dttpgSourcePhContainer_token_;
  edm::EDGetTokenT<L1MuDTChambThContainer> dttpgSourceThContainer_token_;
  edm::InputTag dttpgSource_;

  std::string trstring_;
  edm::EDGetTokenT<L1MuDTTrackContainer> trToken_;
};

#endif
