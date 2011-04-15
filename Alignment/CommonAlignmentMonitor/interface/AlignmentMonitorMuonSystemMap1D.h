#ifndef Alignment_CommonAlignmentMonitor_AlignmentMonitorMuonSystemMap1D_H
#define Alignment_CommonAlignmentMonitor_AlignmentMonitorMuonSystemMap1D_H

/** \class AlignmentMonitorMuonSystemMap1D
 *  $Date: 2011/02/11 23:18:27 $
 *  $Revision: 1.4 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

// system include files
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFromTrack.h"
#include "Alignment/CommonAlignmentMonitor/interface/MuonSystemMapPlot1D.h"

//#include "TTree.h"

// user include files

// 
// class definition
// 

class AlignmentMonitorMuonSystemMap1D: public AlignmentMonitorBase {
public:
  AlignmentMonitorMuonSystemMap1D(const edm::ParameterSet& cfg);
  ~AlignmentMonitorMuonSystemMap1D() {};

  void book();
  void event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& iTrajTracks);
  void afterAlignment(const edm::EventSetup &iSetup);

private:
  friend class MuonSystemMapPlot1D;

  std::string num02d(int num);

  double m_minTrackPt;
  double m_maxTrackPt;
  double m_minTrackP;
  double m_maxTrackP;
  int m_minTrackerHits;
  double m_maxTrackerRedChi2;
  bool m_allowTIDTEC;
  int m_minNCrossedChambers;
  int m_minDT13Hits;
  int m_minDT2Hits;
  int m_minCSCHits;
  bool m_doDT;
  bool m_doCSC;

  MuonSystemMapPlot1D *m_DTvsz_station[4][14]; // [station][sector]
  MuonSystemMapPlot1D *m_CSCvsr_me[2][4][36];  // [endcap][station][chamber]
  MuonSystemMapPlot1D *m_DTvsphi_station[4][5];// [station][wheel]
  MuonSystemMapPlot1D *m_CSCvsphi_me[2][4][3]; // [endcap][station][ring]

  std::vector<MuonSystemMapPlot1D*> m_plots;

  long m_counter_event;
  long m_counter_track;
  long m_counter_trackpt;
  long m_counter_trackp;
  long m_counter_trackokay;
  long m_counter_dt;
  long m_counter_13numhits;
  long m_counter_2numhits;
  long m_counter_csc;
  long m_counter_cscnumhits;

  // optional debug ntuple
  bool m_createNtuple;
  TTree *m_cscnt;

  struct MyCSCDetId
  {
    void init(CSCDetId &id)
    {
      e = id.endcap();
      s = id.station();
      r = id.ring();
      c = id.chamber();
      t = id.iChamberType();
    }
    Short_t e, s, r, c;
    Short_t t; // type 1-10: ME1/a,1/b,1/2,1/3,2/1...4/2
  };
  MyCSCDetId m_id;

  struct MyTrack
  {
    Int_t q;
    Float_t pt, pz;
  };
  MyTrack m_tr;

  struct MyResidual
  {
    Float_t res, slope, rho, phi, z;
  };
  MyResidual m_re;
  
  UInt_t m_run;
};

#endif // Alignment_CommonAlignmentMonitor_AlignmentMonitorMuonSystemMap1D_H
