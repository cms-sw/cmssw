
// -*- C++ -*-
//
// Package:    LumiFileWriter
// Class:      LumiFileWriter
// 
/**\class LumiFileWriter LumiFileWriter.cc RecoLuminosity/LumiFileWriter/src/LumiFileWriter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Adam Hunt
//         Created:  Sun May 11 14:21:30 EDT 2008
// $Id: LumiFileWriter.cc,v 1.11 2009/12/14 22:24:02 wmtan Exp $
//
//

#include "RecoLuminosity/ROOTSchema/interface/LumiFileWriter.hh"
#include "RecoLuminosity/ROOTSchema/interface/ROOTSchema.h"
#include "RecoLuminosity/TCPReceiver/interface/TCPReceiver.h"

#include "RecoLuminosity/TCPReceiver/interface/LumiStructures.hh"

// CMSSW
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
//#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// STL
#include <iostream>

LumiFileWriter::LumiFileWriter(const edm::ParameterSet& iConfig){

   // TCP Receiver configuration
   unsigned int listenPort = iConfig.getUntrackedParameter< unsigned int >("SourcePort", 51002);
   unsigned int AquireMode = iConfig.getUntrackedParameter< unsigned int >("AquireMode",  0);
   DistribIP1 = iConfig.getUntrackedParameter< std::string  >("PrimaryHLXDAQIP", "vmepcS2F17-18");
   DistribIP2 = iConfig.getUntrackedParameter< std::string  >("SecondaryHLXDAQIP", "vmepcS2F17-19");
   reconTime               = iConfig.getUntrackedParameter< unsigned int >("ReconnectionTime",60);

   HLXTCP_ = new HCAL_HLX::TCPReceiver( listenPort, DistribIP1, AquireMode );
   LumiSchema_ = new HCAL_HLX::ROOTSchema();

   // ROOTFileWriter configuration
   std::string lumiFileDir  = iConfig.getUntrackedParameter< std::string  >("LumiFileDir","./");
   std::string lumiFileType = iConfig.getUntrackedParameter< std::string  >("LumiFileType","RAW");

   LumiSchema_->SetLSDir( lumiFileDir );
   LumiSchema_->SetFileType( lumiFileType );

   //ROOTFileMerger configuration
   std::string  MergedOutputDir = iConfig.getUntrackedParameter< std::string  >("MergedOutDir", "./");

   LumiSchema_->SetMergeDir(MergedOutputDir);

   // HTML Generator configuration
   unsigned int NBINS        = iConfig.getUntrackedParameter< unsigned int >("NBINS",     297);  // 12 BX per bin
   double       XMIN         = iConfig.getUntrackedParameter< double       >("XMIN",      0);
   double       XMAX         = iConfig.getUntrackedParameter< double       >("XMAX",      3564);
   std::string  webOutputDir = iConfig.getUntrackedParameter< std::string  >("WBMOutDir", "./");

   LumiSchema_->SetWebDir(webOutputDir);
   LumiSchema_->SetHistoBins( NBINS, XMIN, XMAX );
   
   bMerge_    = iConfig.getUntrackedParameter< bool >("MergeFiles", false );
   bWBM_      = iConfig.getUntrackedParameter< bool >("CreateWebPage", false );
   bTransfer_ = iConfig.getUntrackedParameter< bool >("TransferToDBS", false );

   LumiSchema_->SetMergeFiles( bMerge_ );
   LumiSchema_->SetTransferFiles( bTransfer_ );
   LumiSchema_->SetCreateWebPage( bWBM_ );
   
   lumiSection_ = new HCAL_HLX::LUMI_SECTION;
}

LumiFileWriter::~LumiFileWriter()
{
 
  delete HLXTCP_;
  delete LumiSchema_;
  delete lumiSection_;
}

void LumiFileWriter::analyze(const edm::Event& iEvent, 
			     const edm::EventSetup& iSetup){
  
  while(HLXTCP_->IsConnected() == false){
     HLXTCP_->SetIP( DistribIP1 );
     if( HLXTCP_->Connect() != 1){
	std::cout << "Failed to connect to " << DistribIP1 << "." << std::endl;
	sleep( 1 );
	std::cout << "Trying " << DistribIP2 << std::endl;
	HLXTCP_->SetIP( DistribIP2 );
	if( HLXTCP_->Connect() == 1) break;
	std::cout << "Failed to connect to " << DistribIP2 << "." << std::endl;
	std::cout << " Reconnect in " << reconTime << " seconds." <<  std::endl;
	sleep(reconTime);
     }
  }
  if( HLXTCP_->IsConnected() == true ){
     std::cout << "Successfully connected." << std::endl; 
  }


    
  if( HLXTCP_->ReceiveLumiSection(*lumiSection_) == 1 ){
    
    std::cout << "Processing LumiSection" << std::endl;
    LumiSchema_->ProcessSection(*lumiSection_);
    
  }else{
    HLXTCP_->Disconnect();
    LumiSchema_->EndRun();
  }
}

// ------------ method called once each job just before starting event loop  ------------
void 
LumiFileWriter::beginJob()
{}

// ------------ method called once each job just after ending the event loop  ------------
void 
LumiFileWriter::endJob() {

  HLXTCP_->Disconnect();
  LumiSchema_->EndRun();
}

//define this as a plug-in
DEFINE_FWK_MODULE(LumiFileWriter);
