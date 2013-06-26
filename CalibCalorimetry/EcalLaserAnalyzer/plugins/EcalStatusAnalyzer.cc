/* 
 *  \class EcalStatusAnalyzer
 *
 *  $Date: 2012/02/09 10:07:37 $
 *  author: Julie Malcles - CEA/Saclay
 *  author: Gautier Hamel De Monchenault - CEA/Saclay
 */

#include "EcalStatusAnalyzer.h"

#include "TFile.h"
#include "TTree.h"
#include "TCut.h"
#include "TPaveText.h"
#include "TBranch.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h>
#include <DataFormats/Provenance/interface/Timestamp.h>
using namespace std;

//========================================================================
EcalStatusAnalyzer::EcalStatusAnalyzer(const edm::ParameterSet& iConfig)
//========================================================================
 :
iEvent(0),

// framework parameters with default values
_dataType(        iConfig.getUntrackedParameter< std::string >( "dataType",         "h4"   ) ) // h4 or p5

//========================================================================

{

  //now do what ever initialization is needed

  resdir_                 = iConfig.getUntrackedParameter<std::string>("resDir");
  statusfile_             = iConfig.getUntrackedParameter<std::string>("statusFile");

  eventHeaderCollection_  = iConfig.getParameter<std::string>("eventHeaderCollection");
  eventHeaderProducer_    = iConfig.getParameter<std::string>("eventHeaderProducer");

}


//========================================================================
EcalStatusAnalyzer::~EcalStatusAnalyzer(){
//========================================================================


  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}



//========================================================================
void EcalStatusAnalyzer::beginJob() {
//========================================================================
 

  
  // Initializations
  

  nSM=0;
  fedID=0;
  runType=-999;
  runNum=-999;
  event=0;
 

}


//========================================================================
void EcalStatusAnalyzer:: analyze( const edm::Event & e, const  edm::EventSetup& c){
//========================================================================

  ++iEvent;

  // retrieving DCC header
  edm::Handle<EcalRawDataCollection> pDCCHeader;
  const  EcalRawDataCollection* DCCHeader=0;
  try {
    e.getByLabel(eventHeaderProducer_,eventHeaderCollection_, pDCCHeader);
    DCCHeader=pDCCHeader.product();
  }catch ( std::exception& ex ) {
    std::cerr << "Error! can't get the product  retrieving DCC header " << eventHeaderCollection_.c_str() << std::endl;
  }

  // retrieving TB event header

  edm::Handle<EcalTBEventHeader> pEventHeader; 
  const EcalTBEventHeader* evtHeader=0;
  if ( _dataType == "h4" ){
    try {
      e.getByLabel( eventHeaderProducer_ , pEventHeader );
      evtHeader = pEventHeader.product(); // get a ptr to the product
    } catch ( std::exception& ex ) {
      std::cerr << "Error! can't get the product " << eventHeaderProducer_.c_str() << std::endl;
    }
    
    timeStampCur=evtHeader->begBurstTimeSec();
    nSM=evtHeader->smInBeam();
    
  }
  
  // Get Timestamp 
  
  edm::Timestamp Time=e.time();
  
  if ( _dataType != "h4" ){
    timeStampCur = Time.value();
  }
  
  // ====================================
  // Decode Basic DCCHeader Information 
  // ====================================

  for ( EcalRawDataCollection::const_iterator headerItr= DCCHeader->begin();headerItr != DCCHeader->end(); 
	++headerItr ) {

    // Get run type and run number 
    runType=headerItr->getRunType();
    runNum=headerItr->getRunNumber();
    event=headerItr->getLV1();
    dccID=headerItr->getDccInTCCCommand();
    fedID=headerItr->fedId();  

    short VFEGain=headerItr->getMgpaGain() ;
    short MEMGain=headerItr->getMemGain() ;

   
    // Retrieve laser color and event number
    
    EcalDCCHeaderBlock::EcalDCCEventSettings settings = headerItr->getEventSettings();

    int laser_color = settings.wavelength;
    int laser_power = settings.LaserPower ;
    int laser_filter = settings.LaserFilter ;
    int laser_delay = settings.delay ;
    if(  laser_color <0 ) return;
    // int laser_ = settings.MEMVinj;

    bool isLas=false;
    bool isTP=false;
    bool isPed=false;

    if(runType==EcalDCCHeaderBlock::LASER_STD || runType==EcalDCCHeaderBlock::LASER_GAP
       || runType==EcalDCCHeaderBlock::LASER_POWER_SCAN || runType==EcalDCCHeaderBlock::LASER_DELAY_SCAN) isLas=true;

    else if(runType==EcalDCCHeaderBlock::TESTPULSE_MGPA || runType==EcalDCCHeaderBlock::TESTPULSE_GAP
	    || runType==EcalDCCHeaderBlock::TESTPULSE_SCAN_MEM ) isTP=true;

    else if(runType==EcalDCCHeaderBlock::PEDESTAL_STD  || runType==EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN 
	    || runType==EcalDCCHeaderBlock::PEDESTAL_25NS_SCAN  ) isPed=true;


    // take event only if the fed corresponds to the DCC in TCC
    // and fill gain stuff with value of 1st event
    
    
    if( 600+dccID != fedID ) continue;
    

    bool doesExist=false;
    
    if( (isFedLasCreated.count(fedID)==1 && isLas ) || ( isFedTPCreated.count(fedID)==1 && isTP ) || ( isFedPedCreated.count(fedID)==1 && isPed )) doesExist=true;
    else if(isLas) isFedLasCreated[fedID]=1;
    else if(isTP) isFedTPCreated[fedID]=1;
    else if(isPed) isFedPedCreated[fedID]=1;
    
    
    
    if(!doesExist){
       
      // create new entry for laser

      if (isLas){

	fedIDsLas.push_back(fedID);
	dccIDsLas.push_back(dccID);
	
	timeStampBegLas[fedID]=timeStampCur;
	timeStampEndLas[fedID]=timeStampCur;

	nEvtsLas[fedID]=1;
	runTypeLas[fedID]=runType;

	if (laser_color==iBLUE) {
	  nBlueLas[fedID]=1;
	  laserPowerBlue[fedID] = laser_power;
	  laserFilterBlue[fedID]= laser_filter;
	  laserDelayBlue[fedID] = laser_delay;
	}else if (laser_color==iIR) {
	  nRedLas[fedID]=1;
	  laserPowerRed[fedID] = laser_power;
	  laserFilterRed[fedID]= laser_filter;
	  laserDelayRed[fedID] = laser_delay;
	}

	MGPAGainLas[fedID]=VFEGain;
	MEMGainLas[fedID]=MEMGain;

	  
      }

      // or create new entry for test-pulse
      else if (isTP){
	
	fedIDsTP.push_back(fedID);
	dccIDsTP.push_back(dccID);
	
	nEvtsTP[fedID]=1;
	runTypeTP[fedID]=runType;

	timeStampBegTP[fedID]=timeStampCur;
	timeStampEndTP[fedID]=timeStampCur;

	MGPAGainTP[fedID]=VFEGain;
	MEMGainTP[fedID]=MEMGain;

	// or create new entry for pedestal
	
      } else if (isPed){
	
	fedIDsPed.push_back(fedID);
	dccIDsPed.push_back(dccID);
	
	nEvtsPed[fedID]=1;
	runTypePed[fedID]=runType;
	
	timeStampBegPed[fedID]=timeStampCur;
	timeStampEndPed[fedID]=timeStampCur;
	
	MGPAGainPed[fedID]=VFEGain;
	MEMGainPed[fedID]=MEMGain;
	
      } 

    }else{
      if (isLas){	
	nEvtsLas[fedID]++;
	if (laser_color==iBLUE)nBlueLas[fedID]++;
	else if (laser_color==iIR)nRedLas[fedID]++;
       
	if(timeStampCur<timeStampBegLas[fedID])timeStampBegLas[fedID]=timeStampCur;
	if(timeStampCur>timeStampEndLas[fedID])timeStampEndLas[fedID]=timeStampCur;

      }else if (isTP){	
	nEvtsTP[fedID]++;
	if(timeStampCur<timeStampBegTP[fedID])timeStampBegTP[fedID]=timeStampCur;
	if(timeStampCur>timeStampEndTP[fedID])timeStampEndTP[fedID]=timeStampCur;
      }else if (isPed){	
	nEvtsPed[fedID]++;
	if(timeStampCur<timeStampBegPed[fedID])timeStampBegPed[fedID]=timeStampCur;
	if(timeStampCur>timeStampEndPed[fedID])timeStampEndPed[fedID]=timeStampCur;
      } 
    }  
  }

}// analyze


//========================================================================
void EcalStatusAnalyzer::endJob() {
//========================================================================

  // Create output status file

  std::stringstream namefile;
  namefile << resdir_ <<"/"<<statusfile_;

  string statusfile=namefile.str();
  
  ofstream statusFile(statusfile.c_str(), ofstream::out);
  
  
  if(fedIDsLas.size()!=0 && fedIDsLas.size()==dccIDsLas.size()){
    
    statusFile <<"+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="<<std::endl;
    statusFile <<"                LASER Events              "<<std::endl;
    statusFile <<"+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="<<std::endl;
    
    for(unsigned int i=0;i<fedIDsLas.size();i++){
       
      statusFile <<"RUNTYPE = "<< runTypeLas[fedIDsLas.at(i)]<< std::endl;
      statusFile <<"FEDID = "<< fedIDsLas.at(i) << std::endl;
      statusFile <<"DCCID = "<< dccIDsLas.at(i) << std::endl;
      statusFile <<"TIMESTAMP_BEG = "<<timeStampBegLas[fedIDsLas.at(i)] << std::endl;
      statusFile <<"TIMESTAMP_END = "<<timeStampEndLas[fedIDsLas.at(i)] << std::endl;
      statusFile <<"MPGA_GAIN = "<<MGPAGainLas[fedIDsLas.at(i)] << std::endl;
      statusFile <<"MEM_GAIN  = "<<MEMGainLas[fedIDsLas.at(i)] << std::endl;
      statusFile <<"EVENTS = "<< nEvtsLas[fedIDsLas.at(i)]<< std::endl;
      
      if(nBlueLas[fedIDsLas.at(i)]>0){
	statusFile <<"            blue laser events = "<< nBlueLas[fedIDsLas.at(i)]<< std::endl;
	statusFile <<"            blue laser power  = "<< laserPowerBlue[fedIDsLas.at(i)]<< std::endl;
	statusFile <<"            blue laser filter = "<< laserFilterBlue[fedIDsLas.at(i)]<< std::endl;
	statusFile <<"            blue laser delay  = "<< laserDelayBlue[fedIDsLas.at(i)]<< std::endl;
      }
      
      if(nRedLas[fedIDsLas.at(i)]>0){
	statusFile <<"            ired laser events = "<< nRedLas[fedIDsLas.at(i)]<< std::endl;
	statusFile <<"            ired laser power  = "<< laserPowerRed[fedIDsLas.at(i)]<< std::endl;
	statusFile <<"            ired laser filter = "<< laserFilterRed[fedIDsLas.at(i)]<< std::endl;
	statusFile <<"            ired laser delay  = "<< laserDelayRed[fedIDsLas.at(i)]<< std::endl;
      }
      
      if(i<fedIDsLas.size()-1) statusFile <<"-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="<<std::endl;
      else  statusFile <<"  "<<std::endl;
    }    
  }
  
  if(fedIDsTP.size()!=0 && fedIDsTP.size()==dccIDsTP.size()){
    
    statusFile <<"+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="<<std::endl;
    statusFile <<"             TESTPULSE Events            "<<std::endl;
    statusFile <<"+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="<<std::endl;

    for(unsigned int i=0;i<fedIDsTP.size();i++){
      
      statusFile <<"RUNTYPE = "<< runTypeTP[fedIDsTP.at(i)]<< std::endl;
      statusFile <<"FEDID = "<< fedIDsTP.at(i) << std::endl;
      statusFile <<"DCCID = "<< dccIDsTP.at(i) << std::endl;
      statusFile <<"TIMESTAMP_BEG = "<<timeStampBegTP[fedIDsTP.at(i)] << std::endl;
      statusFile <<"TIMESTAMP_END = "<<timeStampEndTP[fedIDsTP.at(i)] << std::endl;
      statusFile <<"MPGA_GAIN = "<<MGPAGainTP[fedIDsTP.at(i)] << std::endl;
      statusFile <<"MEM_GAIN  = "<<MEMGainTP[fedIDsTP.at(i)] << std::endl;
      statusFile <<"EVENTS = "<< nEvtsTP[fedIDsTP.at(i)]<< std::endl;
      if(i<fedIDsTP.size()-1) statusFile <<"-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="<<std::endl;
      else  statusFile <<"  "<<std::endl;
    }     
  }

  if(fedIDsPed.size()!=0 && fedIDsPed.size()==dccIDsPed.size()){
    
    statusFile <<"+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="<<std::endl;
    statusFile <<"               PEDESTAL Events              "<<std::endl;
    statusFile <<"+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="<<std::endl;

    for(unsigned int i=0;i<fedIDsPed.size();i++){
      
      statusFile <<"RUNTYPE = "<< runTypePed[fedIDsPed.at(i)]<< std::endl;
      statusFile <<"FEDID = "<< fedIDsPed.at(i) << std::endl;
      statusFile <<"DCCID = "<< dccIDsPed.at(i) << std::endl;
      statusFile <<"TIMESTAMP_BEG = "<<timeStampBegPed[fedIDsPed.at(i)] << std::endl;
      statusFile <<"TIMESTAMP_END = "<<timeStampEndPed[fedIDsPed.at(i)] << std::endl;
      statusFile <<"MPGA_GAIN = "<<MGPAGainPed[fedIDsPed.at(i)] << std::endl;
      statusFile <<"MEM_GAIN  = "<<MEMGainPed[fedIDsPed.at(i)] << std::endl;
      statusFile <<"EVENTS = "<< nEvtsPed[fedIDsPed.at(i)]<< std::endl;
      if(i<fedIDsPed.size()-1) statusFile <<"-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="<<std::endl;
      else  statusFile <<"  "<<std::endl;
    }     
  }
  statusFile <<" ... header done"<<std::endl;
  
  statusFile.close();
  
}


DEFINE_FWK_MODULE(EcalStatusAnalyzer);

