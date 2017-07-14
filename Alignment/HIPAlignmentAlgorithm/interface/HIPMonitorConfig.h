#ifndef Alignment_HIPAlignmentAlgorithm_HIPMonitorConfig_h
#define Alignment_HIPAlignmentAlgorithm_HIPMonitorConfig_h

#include <vector>
#include <string>
#include "TTree.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>

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

  void setTree(TTree* tree_){ tree=tree_; }
  void bookBranches(){
    if (tree!=0){
      tree->Branch("Ntracks", &m_Ntracks);
      tree->Branch("Nhits", &m_Nhits);
      //tree->Branch("DataType", &m_datatype); This is done in HIPAlignmentAlgorithm
      tree->Branch("nhPXB", &m_nhPXB);
      tree->Branch("nhPXF", &m_nhPXF);
      tree->Branch("nhTIB", &m_nhTIB);
      tree->Branch("nhTOB", &m_nhTOB);
      tree->Branch("nhTID", &m_nhTID);
      tree->Branch("nhTEC", &m_nhTEC);
      tree->Branch("Pt", &m_Pt);
      tree->Branch("P", &m_P);
      tree->Branch("Eta", &m_Eta);
      tree->Branch("Phi", &m_Phi);
      tree->Branch("Chi2n", &m_Chi2n);
      tree->Branch("d0", &m_d0);
      tree->Branch("dz", &m_dz);
      tree->Branch("wt", &m_wt);
    }
  }
  void fill(){
    if (tree==0) return;
    if (maxNEvents>=0 && nEvents>=maxNEvents) return;

    bool doFill=false;
    int OldSize = m_Ntracks-(int)m_Pt.size();
    if (m_Ntracks==OldSize) return;

    if (maxTracksRcd<0) doFill=true;
    else if (OldSize<maxTracksRcd){
      if (m_Ntracks<maxTracksRcd) doFill=true;
      else{
        int NewSize = maxTracksRcd-OldSize;
        if ((int)m_Pt.size()<NewSize) NewSize=m_Pt.size();

        // Do not touch m_Ntracks, just resize these vectors
        m_Pt.resize(NewSize); m_Eta.resize(NewSize); m_Phi.resize(NewSize); m_Chi2n.resize(NewSize); m_P.resize(NewSize); m_d0.resize(NewSize); m_dz.resize(NewSize); m_wt.resize(NewSize);
        m_Nhits.resize(NewSize); m_nhPXB.resize(NewSize); m_nhPXF.resize(NewSize); m_nhTIB.resize(NewSize); m_nhTOB.resize(NewSize); m_nhTID.resize(NewSize); m_nhTEC.resize(NewSize);

        doFill=true;
      }
    }

    if (doFill){
      tree->Fill();
      nEvents++;
    }

    resetPerEvent();
  }
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
