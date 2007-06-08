#ifndef SiStripMonitorCluster_SiStripMonitorCluster_h
#define SiStripMonitorCluster_SiStripMonitorCluster_h
// -*- C++ -*-
// Package:     SiStripMonitorCluster
// Class  :     SiStripMonitorCluster
/**\class SiStripMonitorCluster SiStripMonitorCluster.h DQM/SiStripMonitorCluster/interface/SiStripMonitorCluster.h
   Data Quality Monitoring source of the Silicon Strip Tracker. Produces histograms related to clusters.
*/
// Original Author:  dkcira
//         Created:  Wed Feb  1 16:47:14 CET 2006
// $Id: SiStripMonitorCluster.h,v 1.10 2006/11/10 17:35:03 dkcira Exp $
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripNoiseService.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class SiStripMonitorCluster : public edm::EDAnalyzer {
  public:
      explicit SiStripMonitorCluster(const edm::ParameterSet&);
      ~SiStripMonitorCluster();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob(edm::EventSetup const&) ;
      virtual void endJob() ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&);
      virtual void endRun(const edm::Run&, const edm::EventSetup&);
      struct ModMEs{ // MEs for one single detector module
        MonitorElement* NumberOfClusters;
        MonitorElement* ClusterPosition;
        MonitorElement* ClusterWidth;
        MonitorElement* ClusterCharge;
        MonitorElement* ClusterSignal;
        MonitorElement* ClusterNoise;
        MonitorElement* ClusterSignalOverNoise;
        MonitorElement* ModuleLocalOccupancy;
        MonitorElement* NrOfClusterizedStrips; // can be used at client level for occupancy calculations
      };
  private:
      void ResetME(MonitorElement* h1);
  private:
       DaqMonitorBEInterface* dbe_;
       edm::ParameterSet conf_;
       SiStripNoiseService SiStripNoiseService_;  
       std::map<uint32_t, ModMEs> ClusterMEs;
       MonitorElement* charge_of_each_cluster;
       // flags
       bool show_mechanical_structure_view, show_readout_view, show_control_view, select_all_detectors, reset_each_run, fill_signal_noise;
};
#endif
