#ifndef L1TDTTPGClient_H
#define L1TDTTPGClient_H
// -*- C++ -*-
//
// Package:    L1TDTTPGClient
// Class:      L1TDTTPGClient
// 
/**\class L1TDTTPGClient L1TDTTPGClient.cc DQM/L1TDTTPGClient/src/L1TDTTPGClient.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lorenzo Agostino
//         Created:  Thu Jun 28 11:32:01 CEST 2007
// $Id: L1TDTTPGClient.h,v 1.2 2007/08/29 16:48:33 lorenzo Exp $
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


class L1TDTTPGClient : public edm::EDAnalyzer, public L1TBaseClient {
   public:
      explicit L1TDTTPGClient(const edm::ParameterSet&);
      ~L1TDTTPGClient();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      virtual void endLuminosityBlock(const edm::LuminosityBlock & l, const edm::EventSetup & c);

      // ----------member data ---------------------------
      int nevents,nupdates;

      DaqMonitorBEInterface *dbe;
      std::string outputFile;
      bool stdalone;
      bool saveOutput;
      bool getMESubscriptionListFromFile;
      bool getQualityTestsFromFile;
      SubscriptionHandle *subscriber;
      QTestHandle * qtHandler;

};

#endif
