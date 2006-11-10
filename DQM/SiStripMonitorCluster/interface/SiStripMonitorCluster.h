#ifndef SiStripMonitorCluster_SiStripMonitorCluster_h
#define SiStripMonitorCluster_SiStripMonitorCluster_h
// -*- C++ -*-
//
// Package:     SiStripMonitorCluster
// Class  :     SiStripMonitorCluster
// 
/**\class SiStripMonitorCluster SiStripMonitorCluster.h DQM/SiStripMonitorCluster/interface/SiStripMonitorCluster.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
// Original Author:  dkcira
//         Created:  Wed Feb  1 16:47:14 CET 2006
// $Id: SiStripMonitorCluster.h,v 1.9 2006/08/07 12:21:36 dkcira Exp $
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

   private:
       struct ModMEs{
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
       DaqMonitorBEInterface* dbe_;
       edm::ParameterSet conf_;
       SiStripNoiseService SiStripNoiseService_;  
       std::map<uint32_t, ModMEs> ClusterMEs;
       MonitorElement* charge_of_each_cluster;
};

#endif
