#ifndef DQMOFFLINE_ECAL_ECALOFFLINECOSMICTASKCLIENT_H
#define DQMOFFLINE_ECAL_ECALOFFLINECOSMICTASKCLIENT_H

#include <memory>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class EcalOfflineCosmicClient : public edm::EDAnalyzer
{
   public:
      explicit EcalOfflineCosmicClient(const edm::ParameterSet& iConfig);
      ~EcalOfflineCosmicClient();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob(const edm::EventSetup&);
      virtual void endJob();
      virtual void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      void end();
      void doProfile3D(MonitorElement* me, std::string name);
      void doProfileX(MonitorElement* me, std::string name);
      void doProfile(std::string path, std::string name);

   private:
      DQMStore *dbe_;
      std::string fileName_;
      bool saveFile_;
      std::string endFunction_;
      std::string rootDir_;
      std::vector<std::string> subDetDirs_;
      std::vector<std::string> l1TriggerDirs_;
      std::string timingDir_;
      std::string timingVsAmp_;
      std::string timingTTBinned_;
      std::string timingModBinned_;
      std::string clusterDir_;
      std::vector<std::string> clusterPlots_;
};

#endif
