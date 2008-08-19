
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
// $Id: LumiFileWriter.cc,v 1.2 2008/08/01 16:30:27 ahunt Exp $
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

#include "RecoLuminosity/TCPReceiver/interface/ICTypeDefs.hh"
#include "RecoLuminosity/TCPReceiver/interface/LumiStructures.hh"

// ROOT Schema Headers
#include "RecoLuminosity/ROOTSchema/interface/ROOTFileReader.h"
#include "RecoLuminosity/ROOTSchema/interface/ROOTFileMerger.h"
#include "RecoLuminosity/ROOTSchema/interface/ROOTFileTransfer.h"

#include "RecoLuminosity/TCPReceiver/interface/TimeStamp.h"
#include "RecoLuminosity/ROOTSchema/interface/FileToolKit.h"

#include "RecoLuminosity/TCPReceiver/interface/TCPReceiver.h"
#include "RecoLuminosity/ROOTSchema/interface/ROOTSchema.h"
#include "RecoLuminosity/ROOTSchema/interface/HTMLGenerator.hh"

#include "RecoLuminosity/ROOTSchema/interface/LumiFileWriter.hh"

LumiFileWriter::LumiFileWriter(const edm::ParameterSet& iConfig){

   // TCP Receiver configuration
   unsigned int listenPort = iConfig.getUntrackedParameter< unsigned int >("SourcePort", 51002);
   unsigned int AquireMode = iConfig.getUntrackedParameter< unsigned int >("AquireMode",  0);
   std::string  DistribIP  = iConfig.getUntrackedParameter< std::string  >("HLXDAQIP",    "vmepcS2F17-19");
   reconTime               = iConfig.getUntrackedParameter< unsigned int >("ReconnectionTime",60);

   HLXTCP.SetPort(listenPort);
   HLXTCP.SetMode(AquireMode);
   HLXTCP.SetIP(DistribIP);

   // ROOTFileWriter configuration
   bool         EtSumOnly       = iConfig.getUntrackedParameter< bool >("EtSumOnly", false);
   std::string lumiFileDir = iConfig.getUntrackedParameter< std::string  >("LumiFileDir","./");

   lumiSchema.SetOutputDir( lumiFileDir );
   lumiSchema.SetEtSumOnly( EtSumOnly );

   //ROOTFileMerger configuration
   std::string  MergedOutputDir = iConfig.getUntrackedParameter< std::string  >("MergedOutDir", "./");

   RFM.SetOutputDir(MergedOutputDir);
   RFM.SetEtSumOnly( EtSumOnly );

   // HTML Generator configuration
   unsigned int NBINS        = iConfig.getUntrackedParameter< unsigned int >("NBINS",     297);  // 12 BX per bin
   double       XMIN         = iConfig.getUntrackedParameter< double       >("XMIN",      0);
   double       XMAX         = iConfig.getUntrackedParameter< double       >("XMAX",      3564);
   std::string  webOutputDir = iConfig.getUntrackedParameter< std::string  >("WBMOutDir", "./");

   webPage.SetOutputDir(webOutputDir);
   webPage.SetHistoBins( NBINS, XMIN, XMAX );
   
   bMerge_    = iConfig.getUntrackedParameter< bool   >("MergeFiles", false );
   bWBM_      = iConfig.getUntrackedParameter< bool   >("CreateWebPage", false );
   MergeRate_ = iConfig.getUntrackedParameter< unsigned int >("MergeRate", 1 );
   bTransfer_ = iConfig.getUntrackedParameter< bool >("TransferToDBS", false );
   bTest_     = iConfig.getUntrackedParameter< bool >("Test", false );

   lastRun_ = 0;

}

LumiFileWriter::~LumiFileWriter()
{
 
}

void LumiFileWriter::analyze(const edm::Event& iEvent, 
			     const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  while(HLXTCP.IsConnected() == false){
    if(HLXTCP.Connect() != 1){
      std::cout << " Reconnect in " << reconTime << " seconds." <<  std::endl;
      sleep(reconTime);
    }
  }
    
  if( HLXTCP.ReceiveLumiSection(localSection) == 1 ){
    
    std::cout << "Writing LS file" << std::endl;
    lumiSchema.ProcessSection(localSection);
    
    lastRun_ = localSection.hdr.runNumber;
    lastCMSLive_  = localSection.hdr.bCMSLive;

    if( bWBM_ ){
      std::cout << "Create Web page" << std::endl; 
      webPage.ReplaceFile(lumiSchema.GetFileName());
      webPage.GetEntry(0);
      webPage.CreateWebPage();
    }
    
  }else{

    HLXTCP.Disconnect();

    if( bMerge_ ){
      if( lastRun_ != 0 ){
	if( LSCount_ % MergeRate_ == 0 ){
	  std::cout << "Merge files" << std::endl;    
	  RFM.Merge( lastRun_ , lastCMSLive_ );
	}
	
	if( bTransfer_ ){
	  std::cout << "Transfer files" << std::endl;    
	  RFT.SetFileName( RFM.GetJustFileName() );
	  RFT.TransferFile( );
	}
	lastRun_ = 0;
      }
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void 
LumiFileWriter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
LumiFileWriter::endJob() {

  if(bTest_){
    HLXTCP.Disconnect();
    
    if( bTransfer_ ){
      std::cout << "Transfer files" << std::endl;    
      RFT.SetFileName( RFM.GetJustFileName() );
      RFT.TransferFile( );
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(LumiFileWriter);
