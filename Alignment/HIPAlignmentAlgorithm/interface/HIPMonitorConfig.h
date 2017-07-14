#ifndef Alignment_HIPAlignmentAlgorithm_HIPMonitorConfig_h
#define Alignment_HIPAlignmentAlgorithm_HIPMonitorConfig_h

#include <vector>
#include <string>
#include "TTree.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <CondFormats/Alignment/interface/Definitions.h>

struct HIPTrackMonitorVariables{
  const int maxNEvents;
  const int maxTracksRcd;
  int nEvents;
  int m_Ntracks;
  std::vector<float> m_Pt, m_Eta, m_Phi, m_Chi2n, m_P, m_d0, m_dz, m_wt;
  std::vector<int> m_Nhits, m_nhPXB, m_nhPXF, m_nhTIB, m_nhTOB, m_nhTID, m_nhTEC;

  TTree* tree;

  HIPTrackMonitorVariables(int maxNEvents_=-1, int maxTracksRcd_=-1) : maxNEvents(maxNEvents_), maxTracksRcd(maxTracksRcd_), nEvents(0), m_Ntracks(0), tree(0){}

  void resetPerEvent(){
    // Do not reset m_Ntracks
    m_Pt.clear(); m_Eta.clear(); m_Phi.clear(); m_Chi2n.clear(); m_P.clear(); m_d0.clear(); m_dz.clear(); m_wt.clear();
    m_Nhits.clear(); m_nhPXB.clear(); m_nhPXF.clear(); m_nhTIB.clear(); m_nhTOB.clear(); m_nhTID.clear(); m_nhTEC.clear();
  }
  void resizeVectors(int NewSize){
    m_Pt.resize(NewSize); m_Eta.resize(NewSize); m_Phi.resize(NewSize); m_Chi2n.resize(NewSize); m_P.resize(NewSize); m_d0.resize(NewSize); m_dz.resize(NewSize); m_wt.resize(NewSize);
    m_Nhits.resize(NewSize); m_nhPXB.resize(NewSize); m_nhPXF.resize(NewSize); m_nhTIB.resize(NewSize); m_nhTOB.resize(NewSize); m_nhTID.resize(NewSize); m_nhTEC.resize(NewSize);
  }

  void setTree(TTree* tree_){ tree=tree_; }
  void bookBranches();
  void fill();

};

struct HIPHitMonitorVariables{
  const int maxHitsRcd;
  int nHits;

  bool m_hasHitProb;
  float m_sinTheta, m_hitwt, m_angle, m_probXY, m_probQ;
  unsigned int m_rawQualityWord;
  align::ID m_detId;

  TTree* tree;

  void resetPerHit(){
    m_hasHitProb=false;
    m_sinTheta=0;
    m_hitwt=1;
    m_angle=0;
    m_probXY=-1;
    m_probQ=-1;
    m_rawQualityWord=9999;
    m_detId=0;
  }

  HIPHitMonitorVariables(int maxHitsRcd_=-1) : maxHitsRcd(maxHitsRcd_), nHits(0), tree(0){ resetPerHit(); }

  void setTree(TTree* tree_){ tree=tree_; }
  void bookBranches();
  void fill();

};

struct HIPMonitorConfig{
  const edm::ParameterSet cfgMonitor;

  const std::string outfilecore;

  const int maxEventsPerJob;

  const bool fillTrackMonitoring;
  const int maxTracks;
  HIPTrackMonitorVariables trackmonitorvars;

  const bool fillTrackHitMonitoring;
  const int maxHits;
  HIPHitMonitorVariables hitmonitorvars;

  std::string outfile;

  int eventCounter;
  int hitCounter;

  HIPMonitorConfig(const edm::ParameterSet& cfg);
  HIPMonitorConfig(const HIPMonitorConfig& other);
  ~HIPMonitorConfig(){}

  bool checkNevents();
  bool checkNhits();
};

#endif
