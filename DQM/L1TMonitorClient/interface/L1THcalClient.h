#ifndef L1THcalClient_H
#define L1THcalClient_H
// -*- C++ -*-
//
// Package:    L1THcalClient
// Class:      L1THcalClient
// 
/**\class L1THcalClient L1THcalClient.cc DQM/L1THcalClient/src/L1THcalClient.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Adam Aurisano
//         Created:  Sun Nov 25 21:32:01 CEST 2007
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
#include "DQMServices/Core/interface/DQMStore.h"
#include <TH1F.h>
#include <TH2F.h>

using namespace std;


#include "DQMServices/Core/interface/MonitorElement.h"
//
// class decleration
class SubscriptionHandle;
class QTestHandle;

class L1THcalClient : public edm::EDAnalyzer{
   public:
      explicit L1THcalClient(const edm::ParameterSet&);
      ~L1THcalClient();
      TH1F * get1DHisto(string meName, DQMStore * dbi);
      TH2F * get2DHisto(string meName, DQMStore * dbi);


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      // virtual void endLuminosityBlock(const edm::LuminosityBlock & l, const edm::EventSetup & c);
      void calcEff(TH1F* num, TH1F* den, MonitorElement* me);

      // ----------member data ---------------------------
      int nevents,nupdates;

      DQMStore *dbe;
      //std::string outputFile;
      //std::string qualityCriterionName;
      std::string input_dir;
      std::string output_dir;
      int minEventsforFit;
      //bool stdalone;
      //bool saveOutput;
      //bool getMESubscriptionListFromFile;
      //bool getQualityTestsFromFile;
      //SubscriptionHandle *subscriber;
      //QTestHandle * qtHandler;
      MonitorElement *hcalPlateau_;
      MonitorElement *hcalThreshold_;
      MonitorElement *hcalWidth_;
      MonitorElement *hcalEff_1_;
      MonitorElement *hcalEff_2_;
      MonitorElement *hcalEff_3_;
      MonitorElement *hcalEff_4_;
      MonitorElement *hcalEff_HBHE[56][72];
      MonitorElement *hcalEff_HF[8][18];
      
};

#endif
