#ifndef Alignment_CommonAlignmentMonitor_AlignmentMonitorMuonSystemMap1D_H
#define Alignment_CommonAlignmentMonitor_AlignmentMonitorMuonSystemMap1D_H

/** \class AlignmentMonitorMuonSystemMap1D
 *  $Date: 2009/08/29 18:18:07 $
 *  $Revision: 1.2 $
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
  int m_minTrackerHits;
  double m_maxTrackerRedChi2;
  bool m_allowTIDTEC;
  int m_minDT13Hits;
  int m_minDT2Hits;
  int m_minCSCHits;

  MuonSystemMapPlot1D *m_DTvsz_station1[12];
  MuonSystemMapPlot1D *m_DTvsz_station2[12];
  MuonSystemMapPlot1D *m_DTvsz_station3[12];
  MuonSystemMapPlot1D *m_DTvsz_station4[14];
  MuonSystemMapPlot1D *m_CSCvsr_me1[2][36];
  MuonSystemMapPlot1D *m_CSCvsr_me2[2][36];
  MuonSystemMapPlot1D *m_CSCvsr_me3[2][36];
  MuonSystemMapPlot1D *m_CSCvsr_me4[2][36];

  MuonSystemMapPlot1D *m_DTvsphi_station1[5];
  MuonSystemMapPlot1D *m_DTvsphi_station2[5];
  MuonSystemMapPlot1D *m_DTvsphi_station3[5];
  MuonSystemMapPlot1D *m_DTvsphi_station4[5];

  MuonSystemMapPlot1D *m_CSCvsphi_me11[2];
  MuonSystemMapPlot1D *m_CSCvsphi_me12[2];
  MuonSystemMapPlot1D *m_CSCvsphi_me13[2];
  MuonSystemMapPlot1D *m_CSCvsphi_me14[2];
  MuonSystemMapPlot1D *m_CSCvsphi_me21[2];
  MuonSystemMapPlot1D *m_CSCvsphi_me22[2];
  MuonSystemMapPlot1D *m_CSCvsphi_me31[2];
  MuonSystemMapPlot1D *m_CSCvsphi_me32[2];
  MuonSystemMapPlot1D *m_CSCvsphi_me41[2];
  MuonSystemMapPlot1D *m_CSCvsphi_me42[2];

  std::vector<MuonSystemMapPlot1D*> m_plots;

  long m_counter_event;
  long m_counter_track;
  long m_counter_trackpt;
  long m_counter_trackokay;
  long m_counter_dt;
  long m_counter_13numhits;
  long m_counter_2numhits;
  long m_counter_csc;
  long m_counter_cscnumhits;
};

#endif // Alignment_CommonAlignmentMonitor_AlignmentMonitorMuonSystemMap1D_H
