
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
// $Id: LumiFileWriter.cc,v 1.1 2008/05/12 21:36:54 ahunt Exp $
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

#include "RecoLuminosity/ROOTSchema/interface/LumiFileWriter.h"

LumiFileWriter::LumiFileWriter(const edm::ParameterSet& iConfig){

   // TCP Receiver configuration
   unsigned int listenPort = iConfig.getUntrackedParameter< unsigned int >("SourcePort", 51002);
   unsigned int AquireMode = iConfig.getUntrackedParameter< unsigned int >("AquireMode",  0);
   std::string  DistribIP  = iConfig.getUntrackedParameter< std::string  >("HLXDAQIP",    "vmepcs2f17-19");
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
    
    if( bMerge_ ){
      if( LSCount_ % MergeRate_ == 0 ){
	std::cout << "Merge files" << std::endl;    
	RFM.Merge(localSection.hdr.runNumber, localSection.hdr.bCMSLive);
      }
    }

    if( bWBM_ ){
      std::cout << "Create Web page" << std::endl; 
      webPage.ReplaceFile(lumiSchema.GetFileName());
      webPage.GetEntry(0);
      webPage.CreateWebPage();
    }
    
  }else{

    HLXTCP.Disconnect();
    
    if( bTransfer_ ){
      std::cout << "Transfer files" << std::endl;    
      RFT.SetFileName( RFM.GetFileName() );
      RFT.TransferFile( );
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
}

//define this as a plug-in
DEFINE_FWK_MODULE(LumiFileWriter);
