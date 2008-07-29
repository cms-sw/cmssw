#ifndef L1TDTTF_H
#define L1TDTTF_H

/*
 * \file L1TDTTF.h
 *
 * $Date: 2008/04/30 08:44:32 $
 * $Revision: 1.6 $
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
  MonitorElement* dttf_p_eta[3][6][12];
  MonitorElement* dttf_p_qual[3][6][12];
  MonitorElement* dttf_p_q[3][6][12];
  MonitorElement* dttf_p_pt[3][6][12];
  MonitorElement* dttf_bx[6][12];
  MonitorElement* dttf_bx_2ndTrack[6][12];
  MonitorElement* dttf_bx_wheel[6];
  //MonitorElement* dttf_bx_wheel_2ndTrack[6];
  MonitorElement* dttf_nTracks_wheel[6];
  //MonitorElement* dttf_nTracks_wheel_2ndTrack[6];
  MonitorElement* dttf_nTrksPerEv[3][6][12];
  MonitorElement* dttf_nTracksPerEvent_wheel[6];
  //MonitorElement* dttf_nTracksPerEvent_wheel_2ndTrack[6];

  MonitorElement* dttf_n2ndTracks_wheel[6];

  MonitorElement* dttf_nTracksPerEvent_integ;
  MonitorElement* dttf_p_phi_integ;
  MonitorElement* dttf_p_eta_integ;
  MonitorElement* dttf_p_pt_integ;
  MonitorElement* dttf_p_qual_integ;
  MonitorElement* dttf_p_q_integ;
  MonitorElement* dttf_bx_integ;
  //MonitorElement* dttf_nTracks_integ;
  //MonitorElement* dttf_phys_phi_integ;
  //MonitorElement* dttf_phys_eta_integ;
  //MonitorElement* dttf_phys_pt_integ;
  MonitorElement* dttf_p_phi_eta_integ;

  MonitorElement* dttf_p_phi_eta_wheel[6];
  MonitorElement* dttf_bx_Summary;
  MonitorElement* dttf_occupancySummary;
  MonitorElement* dttf_occupancySummary_r;
  MonitorElement* dttf_highQual_Summary;
  MonitorElement* dttf_2ndTrack_Summary;

  MonitorElement* dttf_p_phi_integ_2ndTrack;
  MonitorElement* dttf_p_eta_integ_2ndTrack;
  MonitorElement* dttf_p_pt_integ_2ndTrack;
  MonitorElement* dttf_p_qual_integ_2ndTrack;
  MonitorElement* dttf_p_q_integ_2ndTrack;
  MonitorElement* dttf_bx_integ_2ndTrack;
  MonitorElement* dttf_nTracks_integ_2ndTrack;
  //MonitorElement* dttf_nTracksPerEvent_integ_2ndTrack;
  MonitorElement* dttf_p_phi_eta_integ_2ndTrack;
  MonitorElement* dttf_bx_Summary_2ndTrack;
  MonitorElement* dttf_occupancySummary_2ndTrack;
  MonitorElement* dttf_highQual_Summary_2ndTrack;

  MonitorElement* dttpgphmap;
  MonitorElement* dttpgphmap2nd;
  MonitorElement* dttpgphmapcorr;
  MonitorElement* dttpgphbestmap;
  MonitorElement* dttpgphbestmapcorr;


  MonitorElement* dttpgthmaph;
  MonitorElement* dttpgthbestmap;
  MonitorElement* dttpgthbestmaph;


  //int bx0,bxp1,bxn1;
  int nev_; // Number of events processed
  bool dttf_track;
  bool dttf_track_2;
  int nev_dttf; //Number of events with at least one DTTF track
  int nev_dttf_track2; //Number of events with at least one DTTF 2nd track
  int nBx[6][3]; //Number of beam crossings for each wheel
  int nBx_2ndTrack[6][3];//Number of beam crossings for each wheel (2nd track only)
  int nOccupancy_integ[6][12];
  int nOccupancy_integ_2ndTrack[6][12];
  int nOccupancy_integ_phi_eta[64][144];
  int nOccupancy_integ_phi_eta_2ndTrack[64][144];
  int nOccupancy_wheel_phi_eta[64][144][6];
  int nHighQual[6][12]; //Number of high quality tracks
  int nHighQual_2ndTrack[6][12];//Number of high quality 2nd tracks
  int n2ndTrack[6][12]; //Number of 2nd tracks
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;
  edm::InputTag dttpgSource_;
};

#endif
