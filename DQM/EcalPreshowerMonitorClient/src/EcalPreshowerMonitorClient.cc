#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalPreshowerMonitorClient/interface/EcalPreshowerMonitorClient.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESPedestalClient.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESIntegrityClient.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESSummaryClient.h"

using namespace cms;
using namespace edm;
using namespace std;

EcalPreshowerMonitorClient::EcalPreshowerMonitorClient(const edm::ParameterSet& ps) {

   verbose_    = ps.getUntrackedParameter<bool>("verbose", false);
   outputFile_ = ps.getUntrackedParameter<string>("OutputFile","");
   inputFile_  = ps.getUntrackedParameter<string>("InputFile","");
   prefixME_   = ps.getUntrackedParameter<string>("prefixME", "EcalPreshower");
   debug_      = ps.getUntrackedParameter<bool>("debug", false);

   prescaleFactor_ = ps.getUntrackedParameter<int>("prescaleFactor", 1);

   //Initial enabledClients
   enabledClients_.push_back("Integrity");
   enabledClients_.push_back("Pedestal");
   enabledClients_.push_back("Summary");

   enabledClients_ = ps.getUntrackedParameter<vector<string> >("enabledClients", enabledClients_);	

   if ( verbose_ ) {
      cout << " Enabled Clients:";
      for ( unsigned int i = 0; i < enabledClients_.size(); i++ ) {
	 cout << " " << enabledClients_[i];
      }
      cout << endl;
   }

   //Setup Clients
   if ( find(enabledClients_.begin(), enabledClients_.end(), "Integrity" ) != enabledClients_.end() ){
      clients_.push_back( new ESIntegrityClient(ps) );
   }

   if ( find(enabledClients_.begin(), enabledClients_.end(), "Pedestal" ) != enabledClients_.end() ){
      clients_.push_back( new ESPedestalClient(ps) );
   }

   if ( find(enabledClients_.begin(), enabledClients_.end(), "Summary" ) != enabledClients_.end() ){
      clients_.push_back( new ESSummaryClient(ps) );
   }

   if(debug_){
      cout<<"PrescaleFactor = "<<prescaleFactor_<<endl; 
   }
}

EcalPreshowerMonitorClient::~EcalPreshowerMonitorClient() {

   if ( verbose_ ) cout << "Finish EcalPreshowerMonitorClient" << endl;

   for ( unsigned int i=0; i<clients_.size(); i++ ) {
      delete clients_[i];
   }

}

void EcalPreshowerMonitorClient::beginJob() {

   if(debug_){ 
      cout<<"EcalPreshowerMonitorClient: beginJob"<<endl;
   }

   ievt_ = 0;
   jevt_ = 0;

   // get hold of back-end interface

   dqmStore_ = Service<DQMStore>().operator->();

   if ( inputFile_.size() != 0 ) {
     if ( dqmStore_ ) {
       dqmStore_->open(inputFile_);
    }
  }

  for ( unsigned int i=0; i<clients_.size(); i++ ) {
    clients_[i]->beginJob(dqmStore_);
    clients_[i]->setup();
  }
}

void EcalPreshowerMonitorClient::beginRun(void) {

   if (debug_) { 
      cout << "EcalPreshowerMonitorClient: beginRun" << endl;
   }

   jevt_ = 0;

   begin_run_ = true;
   end_run_   = false;

   for ( unsigned int i=0; i<clients_.size(); i++ ) {
      clients_[i]->beginRun();
   }
}

void EcalPreshowerMonitorClient::beginRun(const edm::Run & r, const edm::EventSetup & c) {

   if (debug_) { 
      cout << "EcalPreshowerMonitorClient: beginRun" << endl;
   }

   jevt_ = 0;

   begin_run_ = true;
   end_run_   = false;

   for ( unsigned int i=0; i<clients_.size(); i++ ) {
      clients_[i]->beginRun();
   }
}

void EcalPreshowerMonitorClient::endJob(void) {

   if (debug_) { 
      cout << "EcalPreshowerMonitorClient: endJob, ievt = " << ievt_ << endl;
   }

   if ( ! end_run_ ) {
     this->analyze(); 
     this->endRun();
   }
   
   if ( outputFile_.size() != 0 ) {
     cout<<"Store Result in "<<outputFile_<<endl;
     dqmStore_->save(outputFile_);
   }
   
   for ( unsigned int i=0; i<clients_.size(); i++ ) {
     clients_[i]->endJob();
   }

}

void EcalPreshowerMonitorClient::endRun(void) {

   if(debug_){ 
      cout << "EcalPreshowerMonitorClient: endRun, jevt = " << jevt_ << endl;
   }

   begin_run_ = false;
   end_run_   = true;

   for ( unsigned int i=0; i<clients_.size(); i++ ) {
     clients_[i]->analyze();
     clients_[i]->endRun();
   }
   
}

void EcalPreshowerMonitorClient::endRun(const edm::Run & r, const edm::EventSetup & c) {

   if(debug_){ 
      cout << "EcalPreshowerMonitorClient: endRun, jevt = " << jevt_ << endl;
   }

   begin_run_ = false;
   end_run_   = true;

   for ( unsigned int i=0; i<clients_.size(); i++ ) {
     clients_[i]->analyze();
     clients_[i]->endRun();
   }

}

void EcalPreshowerMonitorClient::analyze(void) {

   if(debug_){ 
      cout << "EcalPreshowerMonitorClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
   }

   for ( unsigned int i=0; i<clients_.size(); i++ ) {
      clients_[i]->analyze();
   }
}

void EcalPreshowerMonitorClient::analyze(const Event & e, const EventSetup & c) {

   ievt_++;
   jevt_++;

   if(debug_) cout<<" analyze(const Event & e, const EventSetup & c) is called"<<endl;

   if ( prescaleFactor_ > 0 ) {
      if ( jevt_ % prescaleFactor_ == 0 ) this->analyze();
   }

}

void EcalPreshowerMonitorClient::beginLuminosityBlock(const edm::LuminosityBlock &l, const edm::EventSetup & c) {
}

void EcalPreshowerMonitorClient::endLuminosityBlock(const edm::LuminosityBlock &l, const edm::EventSetup & c) {

  for ( unsigned int i=0; i<clients_.size(); i++ ) {
    clients_[i]->endLumiAnalyze();
  }
   
}

void EcalPreshowerMonitorClient::htmlOutput(int run) {

   //Change runNum_ into run by Yeong-jyi 

   string border[2][10] = {
      {"style=\"border-top:solid white; border-left:solid white; border-bottom:solid white; border-right:solid white; border-width:1\"",
	 "style=\"border-top:solid black; border-left:solid black; border-bottom:solid white; border-right:solid white; border-width:1\"",
	 "style=\"border-top:solid black; border-left:solid white; border-bottom:solid white; border-right:solid black; border-width:1\"",
	 "style=\"border-top:solid white; border-left:solid black; border-bottom:solid white; border-right:solid white; border-width:1\"",
	 "style=\"border-top:solid white; border-left:solid white; border-bottom:solid white; border-right:solid black; border-width:1\"",
	 "style=\"border-top:solid black; border-left:solid black; border-bottom:solid white; border-right:solid black; border-width:1\"",
	 "style=\"border-top:solid white; border-left:solid black; border-bottom:solid black; border-right:solid black; border-width:1\"",
	 "style=\"border-top:solid white; border-left:solid black; border-bottom:solid black; border-right:solid white; border-width:1\"",
	 "style=\"border-top:solid white; border-left:solid white; border-bottom:solid black; border-right:solid black; border-width:1\"",
	 "style=\"border-top:solid white; border-left:solid black; border-bottom:solid white; border-right:solid black; border-width:1\""},

      {"style=\"border-top:solid white; border-left:solid white; border-bottom:solid white; border-right:solid white; border-width:1\"",
	 "style=\"border-top:solid black; border-left:solid black; border-bottom:solid white; border-right:solid white; border-width:1\"",
	 "style=\"border-top:solid black; border-left:solid white; border-bottom:solid white; border-right:solid black; border-width:1\"",
	 "style=\"border-top:solid black; border-left:solid white; border-bottom:solid white; border-right:solid white; border-width:1\"",
	 "style=\"border-top:solid white; border-left:solid white; border-bottom:solid black; border-right:solid white; border-width:1\"",
	 "style=\"border-top:solid black; border-left:solid black; border-bottom:solid black; border-right:solid white; border-width:1\"",
	 "style=\"border-top:solid black; border-left:solid white; border-bottom:solid black; border-right:solid black; border-width:1\"",
	 "style=\"border-top:solid white; border-left:solid black; border-bottom:solid black; border-right:solid white; border-width:1\"",
	 "style=\"border-top:solid white; border-left:solid white; border-bottom:solid black; border-right:solid black; border-width:1\"",
	 "style=\"border-top:solid black; border-left:solid white; border-bottom:solid black; border-right:solid white; border-width:1\""}
   };

   int iborder[2][40][40] = {
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 1, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 3, 4, 3, 4, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 3, 4, 3, 4, 3, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 3, 4, 7, 8, 3, 4, 7, 8, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 7, 8, 3, 4, 7, 8, 3, 4, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 5, 3, 4, 1, 2, 7, 8, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 1, 2, 7, 8, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 1, 4, 7, 8, 3, 4, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 1, 2, 3, 4, 7, 8, 3, 2, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 5, 3, 4, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 7, 8, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 1, 2, 3, 4, 5, 0, 0, 0, 0},
	 {0, 0, 0, 1, 4, 7, 8, 3, 4, 3, 4, 3, 4, 7, 8, 7, 8, 7, 8, 1, 2, 7, 8, 7, 8, 7, 8, 3, 4, 3, 4, 3, 4, 7, 8, 3, 2, 0, 0, 0},
	 {0, 0, 0, 3, 4, 1, 2, 3, 4, 7, 8, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4, 7, 8, 3, 4, 1, 2, 3, 4, 0, 0, 0},
	 {0, 0, 0, 7, 8, 3, 4, 7, 8, 1, 2, 7, 8, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 7, 8, 1, 2, 7, 8, 3, 4, 7, 8, 0, 0, 0},
	 {0, 0, 5, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 3, 4, 3, 8, 7, 8, 7, 4, 3, 4, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 5, 0, 0},
	 {0, 0, 9, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 7, 8, 3, 8, 6, 0, 0, 0, 0, 6, 7, 4, 7, 8, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 9, 0, 0},
	 {0, 1, 4, 3, 4, 7, 8, 3, 4, 3, 4, 3, 4, 1, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 6, 1, 2, 3, 4, 3, 4, 3, 4, 7, 8, 3, 4, 3, 2, 0},
	 {0, 7, 8, 7, 8, 1, 2, 7, 8, 7, 8, 7, 8, 3, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 4, 7, 8, 7, 8, 7, 8, 1, 2, 7, 8, 7, 8, 0},
	 {0, 1, 2, 1, 2, 3, 4, 1, 2, 1, 2, 1, 2, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 1, 2, 0},
	 {0, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 0},
	 {0, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 0},
	 {0, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 0}, //
	 {0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0}, //
	 {0, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 0},
	 {0, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 0},
	 {0, 7, 8, 7, 8, 3, 4, 7, 8, 7, 8, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 7, 8, 7, 8, 7, 8, 3, 4, 7, 8, 7, 8, 0},
	 {0, 1, 2, 1, 2, 7, 8, 1, 2, 1, 2, 1, 2, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 2, 1, 2, 1, 2, 7, 8, 1, 2, 1, 2, 0},
	 {0, 7, 4, 3, 4, 1, 2, 3, 4, 3, 4, 3, 4, 7, 8, 5, 0, 0, 0, 0, 0, 0, 0, 0, 5, 7, 8, 3, 4, 3, 4, 3, 4, 1, 2, 3, 4, 3, 8, 0},
	 {0, 0, 9, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 1, 2, 3, 2, 5, 0, 0, 0, 0, 5, 1, 4, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 9, 0, 0},
	 {0, 0, 6, 7, 8, 3, 4, 7, 8, 3, 4, 7, 8, 3, 4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 4, 7, 8, 3, 4, 7, 8, 3, 4, 7, 8, 6, 0, 0},
	 {0, 0, 0, 1, 2, 3, 4, 1, 2, 7, 8, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 1, 2, 7, 8, 1, 2, 3, 4, 1, 2, 0, 0, 0},
	 {0, 0, 0, 3, 4, 7, 8, 3, 4, 1, 2, 3, 4, 7, 8, 7, 8, 3, 4, 3, 4, 7, 8, 7, 8, 3, 4, 3, 4, 1, 2, 3, 4, 7, 8, 3, 4, 0, 0, 0},
	 {0, 0, 0, 7, 4, 1, 2, 3, 4, 3, 4, 3, 4, 1, 2, 1, 2, 1, 2, 7, 8, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 1, 2, 3, 8, 0, 0, 0},
	 {0, 0, 0, 0, 6, 3, 4, 7, 8, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 7, 8, 3, 4, 6, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 7, 4, 1, 2, 3, 4, 7, 8, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 7, 8, 3, 4, 1, 2, 3, 8, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 6, 3, 4, 7, 8, 1, 2, 7, 8, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 7, 8, 1, 2, 7, 8, 3, 4, 6, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 7, 8, 3, 4, 3, 4, 3, 4, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 7, 8, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 7, 8, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},

      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 9, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 9, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 1, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 2, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 5, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 6, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 0, 0, 0, 0},
	 {0, 0, 0, 0, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 0, 0, 0, 0},
	 {0, 0, 0, 1, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 2, 0, 0, 0},
	 {0, 0, 5, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 6, 0, 0},
	 {0, 0, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 0, 0},
	 {0, 0, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 0, 0},
	 {0, 1, 3, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 9, 6, 0, 0, 0, 0, 5, 9, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 3, 2, 0},
	 {0, 7, 4, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 4, 8, 0},
	 {1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2},
	 {7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8},
	 {1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2},
	 {7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8},
	 {1, 3, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 3, 2}, // 
	 {7, 4, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 4, 8}, //
	 {1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2},
	 {7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8},
	 {1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2},
	 {7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8},
	 {0, 1, 3, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 3, 2, 0},
	 {0, 7, 4, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 9, 6, 0, 0, 0, 0, 5, 9, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 4, 8, 0},
	 {0, 0, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 0, 0},
	 {0, 0, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 0, 0},
	 {0, 0, 5, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 6, 0, 0},
	 {0, 0, 0, 7, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 8, 0, 0, 0},
	 {0, 0, 0, 0, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 0, 0, 0, 0},
	 {0, 0, 0, 0, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 5, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 3, 3, 6, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 7, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 4, 4, 8, 7, 4, 8, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 0, 5, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 9, 3, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 9, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 7, 4, 4, 8, 7, 4, 4, 8, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      }
   };

   // Make HTML
   if (runtype_ ==3) {

      Char_t run_s[50];
      sprintf(run_s, "/var/www/html/DQM/%08d", run);
      string htmlDir = run_s;
      system(("/bin/mkdir -m 777 -p " + htmlDir).c_str());
      sprintf(run_s, "/var/www/html/DQM/%08d/gif", run);
      string gifDir = run_s;
      system(("/bin/mkdir -m 777 -p " + gifDir).c_str());
      ofstream htmlFile;
      htmlFile.open((htmlDir+"/index.html").c_str(), ios::app);

      // Start to display ES planes 
      int iquad_M[2][40] = {{  5,  7, 10, 11, 13, 13, 14, 15, 16, 17,
	 17, 17, 18, 18, 19, 19, 19, 19, 19, 19,
	 19, 19, 19, 19, 19, 19, 18, 18, 17, 17,
	 17, 16, 15, 14, 13, 13, 11, 10,  7,  5},
	  {  0,  6,  8, 11, 12, 13, 14, 16, 16, 17,
	     18, 18, 18, 19, 19, 20, 20, 20, 20, 20,  
	     20, 20, 20, 20, 20, 19, 19, 18, 18, 18,
	     17, 16, 16, 14, 13, 12, 11,  8,  6,  0}
      };
      int iquad_m[2][40] = {{1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1,  1,  1,  3,  5,  6,  7,  7,  8,  8,
	 8,  8,  7,  7,  6,  5,  3,  1,  1,  1,
	 1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	  {1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	     1,  1,  1,  3,  5,  6,  7,  7,  8,  8,
	     8,  8,  7,  7,  6,  5,  3,  1,  1,  1,
	     1,  1,  1,  1,  1,  1,  1,  1,  1,  1}
      };

      htmlFile << "<table border=\"0\" bordercolor=\"white\" cellspacing=\"0\" style=\"float: left;\">" << endl;
      htmlFile << "<tr align=\"center\">" << endl;
      if (layer_==1) htmlFile << "<td colspan=\"40\"> ES- F <a href=\"../plotAll.php?run="<<run<<"&type="<<runtype_<<"\">(summary plots)</a></td>" << endl;
      else if (layer_==1) htmlFile << "<td colspan=\"40\"> ES- R <a href=\"../plotAll.php?run="<<run<<"&type="<<runtype_<<"\">(summary plots)</a></td>" << endl;

      //string color[6] = {"#C0C0C0", "#00FF00", "#FFFF00", "#F87217", "#FF0000", "#0000FF"};
      for (int i=0; i<40; ++i) {
	 htmlFile << "<tr>" << endl;
	 for (int j=0; j<40; ++j) {
	    if ((iquad_m[layer_-1][i]-1) == 0) {
	       if (j<=(19-iquad_M[layer_-1][i]) || j>(19+iquad_M[layer_-1][i])) 
		  htmlFile << "<td "<<border[layer_-1][iborder[layer_-1][i][j]]<<" > <img src=\"../0.png\" width=12 height=12 border=0> </img></td>" << endl;
	       else {
		  htmlFile << "<td "<<border[layer_-1][iborder[layer_-1][i][j]]<<" ><a href=\"../plot.php?run="<<run<<"&type="<<runtype_<<"&iz="<<senZ_[0]<<"&ip="<<senP_[0]<<"&ix="<<j+1<<"&iy="<<39-i+1<<"&gain="<<gain_<<"&prec="<<precision_<<"\" STYLE=\"text-decoration:none\" target=_blank><img src=\"../"<< qt[j][39-i]<<".png\" width=12 height=12 border=0></img></a></td>" << endl;
	       }
	    } else {
	       if (j>=(19-iquad_m[layer_-1][i]+2) && j<=(19+iquad_m[layer_-1][i]-1)) 
		  htmlFile << "<td "<<border[layer_-1][iborder[layer_-1][i][j]]<<" > <img src=\"../0.png\" width=12 height=12 border=0> </img></td>" << endl;
	       else if (j<=(19-iquad_M[layer_-1][i]) || j>(19+iquad_M[layer_-1][i])) 
		  htmlFile << "<td "<<border[layer_-1][iborder[layer_-1][i][j]]<<" > <img src=\"../0.png\" width=12 height=12 border=0> </img></td>" << endl;
	       else {
		  htmlFile << "<td "<<border[layer_-1][iborder[layer_-1][i][j]]<<" ><a href=\"../plot.php?run="<<run<<"&type="<<runtype_<<"&iz="<<senZ_[0]<<"&ip="<<senP_[0]<<"&ix="<<j+1<<"&iy="<<39-i+1<<"&gain="<<gain_<<"&prec="<<precision_<<"\" STYLE=\"text-decoration:none\" target=_blank><img src=\"../"<< qt[j][39-i]<<".png\" width=12 height=12 border=0></img></a></td>" << endl;
	       }
	    }
	 }
	 htmlFile << "</tr>" <<endl;
      }
      htmlFile << "</table>" <<endl;
      htmlFile << "<table border=\"1\">" <<endl;
      if (runtype_==3) {
	 htmlFile << "<tr><td><img src=\"../2.png\" width=20 height=20 border=0> this sensor is not used for injection </img></td></tr>" << endl;
	 htmlFile << "<tr><td><img src=\"../1.png\" width=20 height=20 border=0> the injection result for this sensor is OK </img></td></tr>" << endl;
	 htmlFile << "<tr><td><img src=\"../4.png\" width=20 height=20 border=0> at least one strip has signal 10% higher than average </img></td></tr>" <<endl;
	 htmlFile << "<tr><td><img src=\"../3.png\" width=20 height=20 border=0> at least one strip has signal 10% lower than average </img></td></tr>" <<endl;
	 htmlFile << "<tr><td><img src=\"../7.png\" width=20 height=20 border=0> all strips have signal lower than "<< qtCriteria<<" ADC </img></td></tr>" <<endl;
	 htmlFile << "<tr><td><img src=\"../6.png\" width=20 height=20 border=0> the order of time sample is wrong </img></td></tr>" <<endl;
	 htmlFile << "<tr><td><img src=\"../5.png\" width=20 height=20 border=0> this sensor is used for data-taking, but does not deliver data  </img></td></tr>" <<endl;
      }
      htmlFile << "</table>" <<endl;
      htmlFile << "</body> " << endl;
      htmlFile << "</html> " << endl;

      htmlFile.close();
   }
}


//define this as a plug-in
DEFINE_FWK_MODULE(EcalPreshowerMonitorClient);
