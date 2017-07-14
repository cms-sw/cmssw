#include <Alignment/HIPAlignmentAlgorithm/interface/HIPMonitorConfig.h>


HIPMonitorConfig::HIPMonitorConfig(const edm::ParameterSet& cfg) :
cfgMonitor(cfg.getParameter<edm::ParameterSet>("monitorConfig")),
outfilecore(cfgMonitor.getParameter<std::string>("outfile")),
maxEventsPerJob(cfgMonitor.getParameter<int>("maxEventsPerJob")),
fillTrackMonitoring(cfgMonitor.getParameter<bool>("fillTrackMonitoring")),
maxTracks(cfgMonitor.getParameter<int>("maxTracks")),
trackmonitorvars(maxEventsPerJob, maxTracks),
fillTrackHitMonitoring(cfgMonitor.getParameter<bool>("fillTrackHitMonitoring")),
maxHits(cfgMonitor.getParameter<int>("maxHits")),
hitmonitorvars(maxHits),
eventCounter(0),
hitCounter(0)
{
  outfile = cfg.getParameter<std::string>("outpath") + outfilecore;
}

HIPMonitorConfig::HIPMonitorConfig(const HIPMonitorConfig& other) :
cfgMonitor(other.cfgMonitor),
outfilecore(other.outfilecore),
maxEventsPerJob(other.maxEventsPerJob),
fillTrackMonitoring(other.fillTrackMonitoring),
maxTracks(other.maxTracks),
fillTrackHitMonitoring(other.fillTrackHitMonitoring),
maxHits(other.maxHits),
outfile(other.outfile),
eventCounter(other.eventCounter),
hitCounter(other.hitCounter)
{}

bool HIPMonitorConfig::checkNevents(){ bool res = (maxEventsPerJob<0 || maxEventsPerJob>eventCounter); eventCounter++; return res; }
bool HIPMonitorConfig::checkNhits(){ bool res = (maxHits<0 || maxHits>hitCounter); hitCounter++; return res; }


void HIPTrackMonitorVariables::bookBranches(){
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
void HIPHitMonitorVariables::bookBranches(){
  if (tree!=0){
    tree->Branch("Id", &m_detId, "Id/i");
    tree->Branch("sinTheta", &m_sinTheta);
    tree->Branch("impactAngle", &m_angle);
    tree->Branch("wt", &m_hitwt);
    tree->Branch("probPresent", &m_hasHitProb);
    tree->Branch("probXY", &m_probXY);
    tree->Branch("probQ", &m_probQ);
    tree->Branch("qualityWord", &m_rawQualityWord);
  }
}
void HIPTrackMonitorVariables::fill(){
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
      resizeVectors(NewSize);

      doFill=true;
    }
  }

  if (doFill){
    tree->Fill();
    nEvents++;
  }

  resetPerEvent();
}
void HIPHitMonitorVariables::fill(){
  if (tree==0) return;

  bool doFill=(maxHitsRcd<0 || nHits<maxHitsRcd);

  if (doFill){
    tree->Fill();
    nHits++;
  }

  resetPerHit();
}
