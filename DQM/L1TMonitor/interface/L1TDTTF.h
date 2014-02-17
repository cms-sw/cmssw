#ifndef L1TDTTF_H
#define L1TDTTF_H

/*
 * \file L1TDTTF.h
 *
 * $Date: 2010/11/01 11:27:53 $
 * $Revision: 1.14 $
 * \author J. Berryhill
 *
 */


// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

//
// class declaration
//


class DQMStore;
class MonitorElement;
class L1MuDTTrackCand;
class L1MuRegionalCand;

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
  void beginJob(void);

  // EndJob
  void endJob(void);

  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
			    edm::EventSetup const& context){};

  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
			  edm::EventSetup const& context){};

 private:


  void fillMEs( std::vector<L1MuDTTrackCand> * trackContainer,
		std::vector<L1MuRegionalCand> & gmtDttfCands );
  void setWheelLabel(MonitorElement *me);
  void setQualLabel(MonitorElement *me, int axis);
  void bookEta( int wh, int & nbins, float & start, float & stop );

  // ----------member data ---------------------------
  edm::InputTag dttpgSource_;
  edm::InputTag gmtSource_ ;
  edm::InputTag muonCollectionLabel_;
  std::string l1tsubsystemfolder_;
  bool online_;
  bool verbose_;
  DQMStore * dbe_;
  std::string outputFile_; //file name for ROOT ouput
  edm::InputTag trackInputTag_;

  MonitorElement* dttf_nTracksPerEvent_wheel[6];
  MonitorElement* dttf_quality_wheel_2ndTrack[6];
  MonitorElement* dttf_quality_summary_wheel_2ndTrack[6];
  MonitorElement* dttf_phi_eta_fine_wheel[6];
  MonitorElement* dttf_phi_eta_coarse_wheel[6];
  MonitorElement* dttf_phi_eta_wheel_2ndTrack[6];
  MonitorElement* dttf_eta_wheel_2ndTrack[6];
  MonitorElement* dttf_phi_wheel_2ndTrack[6];
  MonitorElement* dttf_pt_wheel_2ndTrack[6];
  MonitorElement* dttf_q_wheel_2ndTrack[6];

  MonitorElement* dttf_nTracksPerEv[6][12];
  MonitorElement* dttf_bx[6][12];
  MonitorElement* dttf_bx_2ndTrack[6][12];
  MonitorElement* dttf_qual[6][12];
  MonitorElement* dttf_eta_fine_fraction[6][12];
  MonitorElement* dttf_eta[6][12];
  MonitorElement* dttf_phi[6][12];
  MonitorElement* dttf_pt[6][12];
  MonitorElement* dttf_q[6][12];

  MonitorElement* dttf_nTracksPerEvent_integ;
  MonitorElement* dttf_spare;

  MonitorElement* dttf_gmt_match;
  MonitorElement* dttf_gmt_missed;
  MonitorElement* dttf_gmt_ghost;
  // MonitorElement* dttf_gmt_ghost_phys;

  int nev_; // Number of events processed
  int nev_dttf_; //Number of events with at least one DTTF track
  int nev_dttf_track2_; //Number of events with at least one DTTF 2nd track
  int numTracks[6][12];
};

#endif
