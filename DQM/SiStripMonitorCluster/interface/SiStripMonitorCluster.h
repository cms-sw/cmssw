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
//
// Original Author:  dkcira
//         Created:  Wed Feb  1 16:47:14 CET 2006
// $Id: SiStripMonitorCluster.h,v 1.2 2006/03/08 13:04:15 dkcira Exp $
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

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
        MonitorElement* NrClusters;
        MonitorElement* ClusterPosition;
        MonitorElement* ClusterWidth;
        MonitorElement* ClusterCharge;
       };
       DaqMonitorBEInterface* dbe_;
       edm::ParameterSet conf_;
       std::map<uint32_t, ModMEs> ClusterMEs;
};

#endif
