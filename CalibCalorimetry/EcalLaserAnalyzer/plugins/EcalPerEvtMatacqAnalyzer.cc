/* 
 *  \class EcalPerEvtMatacqAnalyzer
 *
 *  $Date: 2012/02/09 10:07:37 $
 *  primary author: Gautier Hamel De Monchenault - CEA/Saclay
 *  author: Julie Malcles - CEA/Saclay
 */

#include <TFile.h>
#include <TTree.h>
#include <TChain.h>

#include "EcalPerEvtMatacqAnalyzer.h"

#include <sstream>
#include <iostream>
#include <iomanip>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Framework/interface/EventSetup.h>

#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>


using namespace std;

//========================================================================
EcalPerEvtMatacqAnalyzer::EcalPerEvtMatacqAnalyzer(const edm::ParameterSet& iConfig)
  :
  iEvent(0),
  // framework parameters with default values

  _presample(     iConfig.getUntrackedParameter< double       >( "nPresamples",   6.7 ) ),
  _nsamplesaftmax(iConfig.getUntrackedParameter< unsigned int >( "nSamplesAftMax", 80 ) ),
  _nsamplesbefmax(iConfig.getUntrackedParameter< unsigned int >( "nSamplesBefMax", 20 ) ),
  _noiseCut(      iConfig.getUntrackedParameter< unsigned int >( "noiseCut",        7 ) ),
  _parabnbefmax(  iConfig.getUntrackedParameter< unsigned int >( "paraBeforeMax",   8 ) ),
  _parabnaftmax(  iConfig.getUntrackedParameter< unsigned int >( "paraAfterMax",    7 ) ),
  _thres(         iConfig.getUntrackedParameter< unsigned int >( "threshold",      10 ) ),
  _lowlev(        iConfig.getUntrackedParameter< unsigned int >( "lowLevel",       20 ) ),
  _highlev(       iConfig.getUntrackedParameter< unsigned int >( "highLevel",      80 ) ),
  _nevlasers(     iConfig.getUntrackedParameter< unsigned int >( "nEventLaser",   600 ) )

//========================================================================
{


  //now do what ever initialization is needed

  resdir_                 = iConfig.getUntrackedParameter<std::string>("resDir");

  digiCollection_         = iConfig.getParameter<std::string>("digiCollection");
  digiProducer_           = iConfig.getParameter<std::string>("digiProducer");
  
  eventHeaderCollection_  = iConfig.getParameter<std::string>("eventHeaderCollection");
  eventHeaderProducer_    = iConfig.getParameter<std::string>("eventHeaderProducer");

}

//========================================================================
EcalPerEvtMatacqAnalyzer::~EcalPerEvtMatacqAnalyzer(){
//========================================================================

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}



//========================================================================
void EcalPerEvtMatacqAnalyzer::beginJob() {
//========================================================================

    //Initializations

    IsFileCreated=0;
    IsTreeCreated=0;

}


//========================================================================
void EcalPerEvtMatacqAnalyzer:: analyze( const edm::Event & e, const  edm::EventSetup& c){
//========================================================================

  ++iEvent;

  // retrieving MATACQ :
  edm::Handle<EcalMatacqDigiCollection> pmatacqDigi;
  const EcalMatacqDigiCollection* matacqDigi=0;
  try {
    e.getByLabel(digiProducer_,digiCollection_, pmatacqDigi); 
    matacqDigi=pmatacqDigi.product();
  }catch ( std::exception& ex ) {
    std::cerr << "Error! can't get the product " << digiCollection_.c_str() << std::endl;

  }
  
  // retrieving DCC header

  edm::Handle<EcalRawDataCollection> pDCCHeader;
  const  EcalRawDataCollection* DCCHeader=0;
  try {
     e.getByLabel(digiProducer_, pDCCHeader); 
     //e.getByLabel(eventHeaderProducer_,eventHeaderCollection_, pDCCHeader);
    DCCHeader=pDCCHeader.product();
  }catch ( std::exception& ex ) {
    std::cerr << "Error! can't get the product " << eventHeaderCollection_.c_str() << std::endl;
  }


  // ====================================
  // Decode Basic DCCHeader Information 
  // ====================================

  for ( EcalRawDataCollection::const_iterator headerItr= DCCHeader->begin();headerItr != DCCHeader->end(); 
    ++headerItr ) {
    event=headerItr->getLV1();
    EcalDCCHeaderBlock::EcalDCCEventSettings settings = headerItr->getEventSettings(); 

    laser_color = (int) settings.wavelength;
    
    // Get run type and run number 

    runType=headerItr->getRunType();
    runNum=headerItr->getRunNumber();
    
    // Cut on runType
    
    if(runType!=EcalDCCHeaderBlock::LASER_STD && runType!=EcalDCCHeaderBlock::LASER_GAP 
       && runType!=EcalDCCHeaderBlock::LASER_POWER_SCAN && runType!=EcalDCCHeaderBlock::LASER_DELAY_SCAN) return; 
    
    // Define output results files' names
    
    if (IsFileCreated==0){

      stringstream namefile;

      namefile << resdir_ <<"/MatacqPerEvt-Run"<<runNum<<".root";        
      outfile=namefile.str();    
      
      outFile = new TFile(outfile.c_str(),"RECREATE");
      
      
      // declaration of the tree to fill
      
      tree = new TTree("MatacqTree","MatacqTree");

      IsFileCreated=1;
      
    }
  }


// ===========================
// Decode Matacq Information
// ===========================

  int iCh=0;
  double max=0;

  for(EcalMatacqDigiCollection::const_iterator it = matacqDigi->begin(); it!=matacqDigi->end(); ++it){ // Loop on matacq channel 
    
    // 
    const EcalMatacqDigi& digis = *it;
    
    if(digis.size()==0 || iCh>=N_channels) continue; 

    max=0;
    maxsamp=0;
    nsamples=digis.size();

    if(IsTreeCreated==0){
      
      //List of branches
      
      stringstream mat;
      mat << "matacq["<<nsamples<<"]/D"<< endl;
 
      tree->Branch( "event",       &event,        "event/I"          );
      tree->Branch( "laser_color", &laser_color , "laser_color/I"    );
      //      tree->Branch( "nsamples",    &nsamples ,    "nsamples/I"       );
      tree->Branch( "matacq",      &matacq ,       mat.str().c_str() );
      tree->Branch( "maxsamp",     &maxsamp ,     "maxsamp/I"        );
      tree->Branch( "ttrig",       &ttrig ,       "ttrig/D"          );
      
      tree->SetBranchAddress( "event",       &event       );
      tree->SetBranchAddress( "laser_color", &laser_color );
      tree->SetBranchAddress( "matacq",      matacq       ); 
      //      tree->SetBranchAddress( "nsamples",    &nsamples    );
      tree->SetBranchAddress( "maxsamp",     &maxsamp     );
      tree->SetBranchAddress( "ttrig",       &ttrig       );
      
      IsTreeCreated=1;
    }

    ttrig=digis.tTrig();
    
    for(int i=0; i<digis.size(); ++i){ // Loop on matacq samples      
      matacq[i]=digis.adcCount(i);
      if(matacq[i]>max) {
	max=matacq[i];
	maxsamp=i;
      }
    }  
    
    
    iCh++; 
  }
  
  tree->Fill();
  
} // analyze


//========================================================================
void EcalPerEvtMatacqAnalyzer::endJob() 
{
  
  tree->Write();
  outFile->Close();


}

DEFINE_FWK_MODULE(EcalPerEvtMatacqAnalyzer);

