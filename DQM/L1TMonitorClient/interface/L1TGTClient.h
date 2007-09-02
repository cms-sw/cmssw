#ifndef L1TGTClient_H
#define L1TGTClient_H
// -*- C++ -*-
//
// Package:    L1TGTClient
// Class:      L1TGTClient
// 
/**\class L1TGTClient L1TGTClient.cc DQM/L1TGTClient/src/L1TGTClient.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lorenzo Agostino
//         Created:  Thu Jun 28 11:32:01 CEST 2007
// $Id: L1TGTClient.h,v 1.1 2007/08/29 16:48:00 lorenzo Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

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


class L1TGTClient : public edm::EDAnalyzer, public L1TBaseClient {
   public:
      explicit L1TGTClient(const edm::ParameterSet&);
      ~L1TGTClient();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      virtual void endLuminosityBlock(const edm::LuminosityBlock & l, const edm::EventSetup & c);

      // ----------member data ---------------------------
      int nevents,nupdates;

      DaqMonitorBEInterface *dbe;
      std::string outputFile;
      std::string qualityCriterionName;
      bool stdalone;
      bool saveOutput;
      bool getMESubscriptionListFromFile;
      bool getQualityTestsFromFile;
      SubscriptionHandle *subscriber;
      QTestHandle * qtHandler;
      MonitorElement *normGTFEBx;

};

#endif
