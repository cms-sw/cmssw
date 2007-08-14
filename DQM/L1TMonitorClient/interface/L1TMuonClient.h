#ifndef L1TMuonClient_H
#define L1TMuonClient_H
// -*- C++ -*-
//
// Package:    L1TMuonClient
// Class:      L1TMuonClient
// 
/**\class L1TMuonClient L1TMuonClient.cc DQM/L1TMuonClient/src/L1TMuonClient.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lorenzo Agostino
//         Created:  Thu Jun 28 11:32:01 CEST 2007
// $Id: L1TMuonClient.h,v 1.1 2007/08/14 15:44:20 lorenzo Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "DQM/L1TMonitorClient/interface/L1TBaseClient.h"

//#include "DQMServices/Core/interface/MonitorUserInterface.h"
//
// class decleration
class SubscriptionHandle;
class QTestHandle;//


class L1TMuonClient : public edm::EDAnalyzer, public L1TBaseClient {
   public:
      explicit L1TMuonClient(const edm::ParameterSet&);
      ~L1TMuonClient();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      int nevents,nupdates;

      DaqMonitorBEInterface *dbe;
      std::string outputFile;
      string ptCriterionName;
      string phiCriterionName;
      string qualityCriterionName;
      bool stdalone;
      bool saveOutput;
      char hname[30];
      vector<float> meanfit;
      MonitorElement *PtTestBadChannels;
      MonitorElement *PhiTestBadChannels;
      MonitorElement *QualTestBadChannels;
      MonitorElement *QualTestBadChannels_;
      MonitorElement *testBadChannels_;
      MonitorElement *MeanFitME;
      MonitorElement *newME;
      MonitorElement *gausExample;
      MonitorElement *gausFitExample;
      bool getMESubscriptionListFromFile;
      bool getQualityTestsFromFile;
      SubscriptionHandle *subscriber;
      QTestHandle * qtHandler;

};

#endif
