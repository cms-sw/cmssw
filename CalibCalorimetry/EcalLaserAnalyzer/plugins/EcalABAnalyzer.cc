/* 
 *  \class EcalABAnalyzer
 *
 *  $Date: 2010/04/09 14:36:32 $
 *  primary author: Julie Malcles - CEA/Saclay
 *  author: Gautier Hamel De Monchenault - CEA/Saclay
 */

#include <TAxis.h>
#include <TH1.h>
#include <TProfile.h>
#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <TMath.h>
#include <TF1.h>

#include "CalibCalorimetry/EcalLaserAnalyzer/plugins/EcalABAnalyzer.h"

#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/EventSetup.h>

#include <Geometry/EcalMapping/interface/EcalElectronicsMapping.h>
#include <Geometry/EcalMapping/interface/EcalMappingRcd.h>

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalDetId/interface/EcalElectronicsId.h>
#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>

#include <vector>

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TShapeAnalysis.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TPNFit.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMom.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TAPD.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TAPDPulse.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/ME.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/MEGeom.h>


//========================================================================
EcalABAnalyzer::EcalABAnalyzer(const edm::ParameterSet& iConfig)
//========================================================================
  :
iEvent(0),

// Framework parameters with default values
 
_type    (        iConfig.getUntrackedParameter< std::string  >( "type",          "LASER" ) ), 

// COMMON:
_nsamples(        iConfig.getUntrackedParameter< unsigned int >( "nSamples",           10 ) ),
_presample(       iConfig.getUntrackedParameter< unsigned int >( "nPresamples",         3 ) ),
_firstsample(     iConfig.getUntrackedParameter< unsigned int >( "firstSample",         1 ) ),
_lastsample(      iConfig.getUntrackedParameter< unsigned int >( "lastSample",          2 ) ),
_timingcutlow(    iConfig.getUntrackedParameter< unsigned int >( "timingCutLow",        2 ) ),
_timingcuthigh(   iConfig.getUntrackedParameter< unsigned int >( "timingCutHigh",       9 ) ),
_timingquallow(   iConfig.getUntrackedParameter< unsigned int >( "timingQualLow",       3 ) ),
_timingqualhigh(  iConfig.getUntrackedParameter< unsigned int >( "timingQualHigh",      8 ) ),
_ratiomincutlow(  iConfig.getUntrackedParameter< double       >( "ratioMinCutLow",    0.4 ) ),
_ratiomincuthigh( iConfig.getUntrackedParameter< double       >( "ratioMinCutHigh",  0.95 ) ),
_ratiomaxcutlow(  iConfig.getUntrackedParameter< double       >( "ratioMaxCutLow",    0.8 ) ),
_pulsemaxcutlow(  iConfig.getUntrackedParameter< double       >( "pulseMaxCutLow",  100.0 ) ),
_pulsemaxcuthigh( iConfig.getUntrackedParameter< double       >( "pulseMaxCutHigh", 15000.0 ) ),
_qualpercent(     iConfig.getUntrackedParameter< double       >( "qualPercent",       0.2 ) ),
_presamplecut(    iConfig.getUntrackedParameter< double       >( "presampleCut",      5.0 ) ),
_ecalPart(        iConfig.getUntrackedParameter< std::string  >( "ecalPart",         "EB" ) ),
_fedid(           iConfig.getUntrackedParameter< int          >( "fedID",            -999 ) ),
_debug(           iConfig.getUntrackedParameter< int          >( "debug",              0  ) ),

// LASER/LED AB:
_fitab(           iConfig.getUntrackedParameter< bool         >( "fitAB",           true  ) ),

_noise(           iConfig.getUntrackedParameter< double       >( "noise",             2.0 ) ),
_chi2cut(         iConfig.getUntrackedParameter< double       >( "chi2cut",          10.0 ) ), 


// LASER/LED AB:
_alpha(           iConfig.getUntrackedParameter< double       >( "alpha",       1.5076494 ) ),
_beta(            iConfig.getUntrackedParameter< double       >( "beta",        1.5136036 ) ),
_nevtmax(         iConfig.getUntrackedParameter< unsigned int >( "nEvtMax",           200 ) ),

// COMMON:
nCrys(                                                                               NCRYSEB),
nPNPerMod(                                                                         NPNPERMOD),
nMod(                                                                                 NMODEE),
nSides(                                                                               NSIDES),
runType(-1), runNum(0), moduleID(-1), fedID(-1), dccID(-1), side(2),
lightside(2), iZ(1), channelIteratorEE(0),phi(-1), eta(-1), event(0), color(-1)
//========================================================================

{

  // Initialization from cfg file

  resdir_                 = iConfig.getUntrackedParameter<std::string>("resDir");
  alphainitpath_          = iConfig.getUntrackedParameter<std::string>("alphaPath", "/nfshome0/ecallaser/calibdata/140110/alphabeta");
  digiCollection_         = iConfig.getParameter<std::string>("digiCollection");
  digiPNCollection_       = iConfig.getParameter<std::string>("digiPNCollection");
  digiProducer_           = iConfig.getParameter<std::string>("digiProducer");

  eventHeaderCollection_  = iConfig.getParameter<std::string>("eventHeaderCollection");
  eventHeaderProducer_    = iConfig.getParameter<std::string>("eventHeaderProducer");

  // Geometrical constants initialization 

  if (_ecalPart == "EB") {
    nCrys    = NCRYSEB;
  } else if (_ecalPart == "EE") {
    nCrys    = NCRYSEE;
  }else{
    cout << "Error: Wrong ecalpart "<<_ecalPart<<", should be EB or EE"<< endl;
    cout << " Setting ecalpart to EB "<<endl;
    _ecalPart="EB";
  }

  if( _type != "LASER" && _type!= "LED" ){
    cout << "Error: Wrong type "<<_type<<", should be LASER or LED"<< endl;
    cout << " Setting type to LASER "<<endl;
    _type="LASER";
  }

  iZ         =  1;
  if(_fedid <= 609 ) 
    iZ       = -1;
  modules    = ME::lmmodFromDcc(_fedid);
  nMod       = modules.size(); 
  nRefChan   = NREFCHAN;
  
  for(unsigned int j=0;j<nCrys;j++){
    wasTimingOK[j]=true;
    wasGainOK[j]=true;
    wasSignalOK[j]=true;
  }
  
  // Quality check flags
  
  isGainOK=true;
  isTimingOK=true;
  isSignalOK=true;

  // Objects dealing with pulses

  APDPulse = new TAPDPulse(_nsamples, _presample, _firstsample, _lastsample, 
			   _timingcutlow, _timingcuthigh, _timingquallow, _timingqualhigh,
			   _ratiomincutlow,_ratiomincuthigh, _ratiomaxcutlow,
			   _pulsemaxcutlow, _pulsemaxcuthigh);
 
 
  // Objects needed for npresample calculation
  Delta01=new TMom();
  Delta12=new TMom();

}

//========================================================================
EcalABAnalyzer::~EcalABAnalyzer(){
  //========================================================================

  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)

}



//========================================================================
void EcalABAnalyzer::beginJob() {
  //========================================================================

  if(!_fitab) return;
  
  // Create temporary files and trees to save adc samples
  //======================================================

  
  for (unsigned int i=0;i<nCrys;i++){
    nevtAB[i]=0 ;
  } 
  
  if ( _debug == 1 ) cout << "-- debug EcalABAnalyzer --  beginJob - _type = "<<_type<< endl;
  
  // Good type events counter
  
  typeEvents=0;
  
  if ( _debug == 1 ) cout << "-- debug EcalABAnalyzer -- beginJobAB"<< endl;
  
  // AlphaBeta files
  
  doesABInitTreeExist=true;
  
  // Check ab file exists
  //=======================
  stringstream nameabfile;  
  nameabfile << resdir_ <<"/AB"<<_fedid <<"_"<<_type<<".root";
  alphafile=nameabfile.str();
  

  // Look at initialisation file
  //==================================
  
  stringstream nameabinitfile;
  //  nameabinitfile << alphainitpath_ <<"/AB"<<_fedid <<"_"<<_type<<".root";
  nameabinitfile << alphainitpath_ <<"/AB"<<_fedid <<"_LASER.root";// FIXME NO INIT FILE FOR LED YET
  alphainitfile=nameabinitfile.str();
  
  FILE *testinit;
  testinit = fopen(alphainitfile.c_str(),"r");   
  if(!testinit) {
    doesABInitTreeExist=false;
  }else{
    fclose(testinit);
  }
  
  if ( _debug == 1 ) cout << "-- debug EcalABAnalyzer -- beginJobAB doesABInit2 "<< doesABInitTreeExist<<"  " <<alphainitfile<<endl;
  
  if( doesABInitTreeExist){ 
    TFile fABInit(nameabinitfile.str().c_str());
    if( ! fABInit.IsZombie() ){
      TTree *ABInit = (TTree*) fABInit.Get("ABCol0");
      
      // 2) Shape analyzer
      
      if( ABInit && ABInit->GetEntries()!=0){
	if ( _debug == 1 ) cout << "-- debug EcalABAnalyzer -- beginJobAB here1"<<endl;
	for(unsigned int ic=0;ic<nColor;ic++){
	  shapana[ic]= new TShapeAnalysis(ABInit, _alpha, _beta, 5.5, 1.0, ic);
	}
      }
      fABInit.Close();
    }
  }else{
    if ( _debug == 1 ) cout << "-- debug EcalABAnalyzer -- beginJobAB here2"<<endl;
    for(unsigned int ic=0;ic<nColor;ic++){
      shapana[ic]= new TShapeAnalysis(_alpha, _beta, 5.5, 1.0, ic);
    }
  }  
  
  for(unsigned int ic=0;ic<nColor;ic++){
    shapana[ic] -> set_const(_nsamples,_firstsample,_lastsample,
			     _presample, _nevtmax, _noise, _chi2cut);
  }

}

//========================================================================
void EcalABAnalyzer:: analyze( const edm::Event & e, const  edm::EventSetup& c){
//========================================================================

  if(!_fitab) return;

  ++iEvent;
  
  // retrieving DCC header

  edm::Handle<EcalRawDataCollection> pDCCHeader;
  const  EcalRawDataCollection* DCCHeader=0;
  try {
    e.getByLabel(eventHeaderProducer_,eventHeaderCollection_, pDCCHeader);
    DCCHeader=pDCCHeader.product();
  }catch ( std::exception& ex ) {
    std::cerr << "Error! can't get the product retrieving DCC header" << 
      eventHeaderCollection_.c_str() <<" "<< eventHeaderProducer_.c_str() << std::endl;
  }
 
  //retrieving crystal data from Event

  edm::Handle<EBDigiCollection>  pEBDigi;
  const  EBDigiCollection* EBDigi=0;
  edm::Handle<EEDigiCollection>  pEEDigi;
  const  EEDigiCollection* EEDigi=0;


  if (_ecalPart == "EB") {
    try {
      e.getByLabel(digiProducer_,digiCollection_, pEBDigi); 
      EBDigi=pEBDigi.product(); 
    }catch ( std::exception& ex ) {
      std::cerr << "Error! can't get the product retrieving EB crystal data " << 
	digiCollection_.c_str() << std::endl;
    } 
  } else if (_ecalPart == "EE") {
    try {
      e.getByLabel(digiProducer_,digiCollection_, pEEDigi); 
      EEDigi=pEEDigi.product(); 
    }catch ( std::exception& ex ) {
      std::cerr << "Error! can't get the product retrieving EE crystal data " << 
	digiCollection_.c_str() << std::endl;
    } 
  } else {
    cout <<" Wrong ecalPart in cfg file " << endl;
    abort();
  }
  
  // retrieving crystal PN diodes from Event
  
  edm::Handle<EcalPnDiodeDigiCollection>  pPNDigi;
  const  EcalPnDiodeDigiCollection* PNDigi=0;
  try {
    e.getByLabel(digiProducer_, pPNDigi);
    PNDigi=pPNDigi.product(); 
  }catch ( std::exception& ex ) {
    std::cerr << "Error! can't get the product " << digiPNCollection_.c_str() << std::endl;
  }

  // retrieving electronics mapping

  edm::ESHandle< EcalElectronicsMapping > ecalmapping;
  const EcalElectronicsMapping* TheMapping=0; 
  try{
    c.get< EcalMappingRcd >().get(ecalmapping);
    TheMapping = ecalmapping.product();
  }catch ( std::exception& ex ) {
    std::cerr << "Error! can't get the product EcalMappingRcd"<< std::endl;
  }


  // ============================
  // Decode DCCHeader Information 
  // ============================
  
  for ( EcalRawDataCollection::const_iterator headerItr= DCCHeader->begin();
	headerItr != DCCHeader->end(); ++headerItr ) {
    
    // Get run type and run number 

    int fed = headerItr->fedId();  
    if(fed!=_fedid && _fedid!=-999) continue; 
    
    runType=headerItr->getRunType();
    runNum=headerItr->getRunNumber();
    event=headerItr->getLV1();

    dccID=headerItr->getDccInTCCCommand();
    fedID=headerItr->fedId();  
    lightside=headerItr->getRtHalf();

    // Check fed corresponds to the DCC in TCC
    
    if( 600+dccID != fedID ) continue;

    // Cut on runType

    if( _type == "LASER" 
	&& runType!=EcalDCCHeaderBlock::LASER_STD 
	&& runType!=EcalDCCHeaderBlock::LASER_GAP 
	&& runType!=EcalDCCHeaderBlock::LASER_POWER_SCAN
	&& runType!=EcalDCCHeaderBlock::LASER_DELAY_SCAN ) return; 
    else if( _type == "LED" 
	     && runType!=EcalDCCHeaderBlock::LED_STD  
	     && runType!=EcalDCCHeaderBlock::LED_GAP ) return; 


    // Retrieve laser color and event number
    
    EcalDCCHeaderBlock::EcalDCCEventSettings settings = headerItr->getEventSettings();
    color = settings.wavelength;
    if( color<0 ) return;
    
    vector<int>::iterator iter = find( colors.begin(), colors.end(), color );
    if( iter==colors.end() ){
      colors.push_back( color );
    }
  }

  // Cut on fedID
  
  if(fedID!=_fedid && _fedid!=-999) return; 
  
  // Count good type events 

  typeEvents++;


  // ===========================
  // Decode EBDigis Information
  // ===========================

  adcGain=0;

  // Get Back Color number for shapana
  //==================================
  unsigned int iCol=0;
  for(unsigned int i=0;i<colors.size();i++){
    if(color==colors[i]) {
      iCol=i;
      i=colors.size();
    }
  }
  
  if (EBDigi){
    
    // Loop on crystals
    //=================== 
    
    for ( EBDigiCollection::const_iterator digiItr= EBDigi->begin();
	  digiItr != EBDigi->end(); ++digiItr ) { 
      
      // Retrieve geometry
      //=================== 
      
      EBDetId id_crystal(digiItr->id()) ;
      EBDataFrame df( *digiItr );
      EcalElectronicsId elecid_crystal = TheMapping->getElectronicsId(id_crystal);
      
      int etaG = id_crystal.ieta() ;  // global
      int phiG = id_crystal.iphi() ;  // global
      
      std::pair<int, int> LocalCoord=MEEBGeom::localCoord( etaG , phiG );
      
      int etaL=LocalCoord.first ; // local
      int phiL=LocalCoord.second ;// local
      
      int strip=elecid_crystal.stripId();
      int xtal=elecid_crystal.xtalId();
      
      int module= MEEBGeom::lmmod(etaG, phiG);
      int tower=elecid_crystal.towerId();
            
      unsigned int channel=MEEBGeom::electronic_channel( etaL, phiL );

      assert( channel <= nCrys );

      if( nevtAB[channel] >= _nevtmax ) continue;

      side=MEEBGeom::side(etaG,phiG);

      if( _type == "LASER" && lightside!=side) continue;
      
      assert( module>=*min_element(modules.begin(),modules.end()) && module<=*max_element(modules.begin(),modules.end()) );
      
      eta = etaG;
      phi = phiG;
      channelID=5*(strip-1) + xtal-1; 
      towerID=tower;
  
      // APD Pulse
      //=========== 

      // Loop on adc samples 
     
      for (unsigned int i=0; i< (*digiItr).size() ; ++i ) {   

	EcalMGPASample samp_crystal(df.sample(i));
	adc[i]=samp_crystal.adc() ;    
	adcG[i]=samp_crystal.gainId();   
	adc[i]*=adcG[i];
	if (i==0) adcGain=adcG[i];
	if (i>0) adcGain=TMath::Max(adcG[i],adcGain);  
      }

      APDPulse->setPulse(adc);
      
      // Quality checks
      //================
      
      if(adcGain!=1){
	nEvtBadGain[channel]++;   
	continue;
      }
   
      if(!APDPulse->isTimingQualOK()) nEvtBadTiming[channel]++;
      if(!APDPulse->isPulseOK()) nEvtBadSignal[channel]++;
      nEvtTot[channel]++;

      // Fill if Pulse is fine
      //=======================

      
      if( APDPulse->isPulseOK() ){
	
	Delta01->addEntry(APDPulse->getDelta(0,1));
	Delta12->addEntry(APDPulse->getDelta(1,2));
	
	if( nevtAB[channel] < _nevtmax ){
	  shapana[iCol] -> putAllVals(channel, adc, eta, phi, dccID, side, towerID, channelID);
   	  nevtAB[channel]++ ;
	}
      }
    }
    
  } else if (EEDigi) {

    // Loop on crystals
    //===================
    
    for ( EEDigiCollection::const_iterator digiItr= EEDigi->begin(); 
	  digiItr != EEDigi->end(); ++digiItr ) {  
      
      // Retrieve geometry
      //=================== 
      
      EEDetId id_crystal(digiItr->id()) ;
      EEDataFrame df( *digiItr );
      EcalElectronicsId elecid_crystal = TheMapping->getElectronicsId(id_crystal);
       
      int phiG = id_crystal.iy() ; 
      int etaG = id_crystal.ix() ;

      int iX = (etaG-1)/5+1;
      int iY = (phiG-1)/5+1;
      
      side=MEEEGeom::side(iX, iY, iZ);

      int tower=elecid_crystal.towerId();
      int ch=elecid_crystal.channelId()-1; 

      int module=MEEEGeom::lmmod( iX, iY );
      if( module>=18 && side==1 ) module+=2;

      int hashedIndex=100000*etaG+phiG;
      if( channelMapEE.count(hashedIndex) == 0 ){
	channelMapEE[hashedIndex]=channelIteratorEE;
	channelIteratorEE++;
      }
      unsigned int channel=channelMapEE[hashedIndex];

      if( nevtAB[channel] >= _nevtmax ) continue;


      if( _type == "LASER" && lightside!=side) continue;
      
      assert( module>=*min_element(modules.begin(),modules.end()) 
	      && module<=*max_element(modules.begin(),modules.end()) );
      
      eta = etaG;
      phi = phiG;
      channelID=ch;
      towerID=tower;
      
      assert ( channel < nCrys );
      
      if ( _debug == 2 ) cout << "-- debug EcalABAnalyzer -- analyze: in EEDigi - towerID:"<< towerID<<
		       " channelID:" <<channelID<<" module:"<< module<<
		       " modules:"<<modules.size()<< endl;
      
      // APD Pulse
      //=========== 

      if( (*digiItr).size()>10) cout <<"SAMPLES SIZE > 10!" <<  (*digiItr).size()<< endl;
 
      // Loop on adc samples  

      for (unsigned int i=0; i< (*digiItr).size() ; ++i ) { 

	EcalMGPASample samp_crystal(df.sample(i));
	adc[i]=samp_crystal.adc() ;    
	adcG[i]=samp_crystal.gainId();   
	adc[i]*=adcG[i];
	
	if (i==0) adcGain=adcG[i];
	if (i>0) adcGain=TMath::Max(adcG[i],adcGain);  
      }
      
      APDPulse->setPulse(adc);
      
      // Quality checks
      //================
      
      if(adcGain!=1){ 
	nEvtBadGain[channel]++;   
	continue;
      }
      if(!APDPulse->isTimingQualOK()) nEvtBadTiming[channel]++;
      if(!APDPulse->isPulseOK()) nEvtBadSignal[channel]++;
      nEvtTot[channel]++;

      
      // Fill if Pulse is fine
      //=======================
      

      if( APDPulse->isPulseOK()){
	
	Delta01->addEntry(APDPulse->getDelta(0,1));
	Delta12->addEntry(APDPulse->getDelta(1,2));
	
	if( nevtAB[channel] < _nevtmax ){
	  shapana[iCol] -> putAllVals(channel, adc, eta, phi, dccID, side, towerID, channelID); 
	  nevtAB[channel]++ ;
	}
      }
    }
  }
}

void EcalABAnalyzer::endJob() {

  if(!_fitab) return;
  
  // Don't do anything if there is no events
  //=========================================

  if( typeEvents == 0 ){

    cout << "\n\t***  No "<< _type<< " Events  ***"<< endl;
    return;
  }
  
  if ( _debug == 1 ) cout << "-- debug EcalABAnalyzer -- endJob - still Here"<< endl;
  
  // Adjust channel numbers for EE 
  //===============================

  if( _ecalPart == "EE" ) {
    nCrys=channelMapEE.size();
  }
  if ( _debug == 1 ) cout << "-- debug EcalABAnalyzer -- endJob - nCrys = "<< nCrys << endl;
  
  // Set presamples number 
  //======================
  
  double delta01=Delta01->getMean();
  double delta12=Delta12->getMean();
  if(delta12>_presamplecut) {
    _presample=2;
    if(delta01>_presamplecut) _presample=1;
  }
  
  APDPulse->setPresamples(_presample);
  
  if ( _debug == 1 ) cout << "-- debug EcalABAnalyzer -- endJob - _presample = "<< _presample << endl;


  // Set quality flags for gains, timing, signal
  //=============================================

  double BadGainEvtPercentage=0.0;
  double BadTimingEvtPercentage=0.0;
  double BadSignalEvtPercentage=0.0;
  
  int nChanBadGain=0;
  int nChanBadTiming=0;
  int nChanBadSignal=0;
  
  for (unsigned int i=0;i<nCrys;i++){
    
    if(nEvtTot[i]!=0){
      BadGainEvtPercentage=double(nEvtBadGain[i])/double(nEvtTot[i]);
      BadTimingEvtPercentage=double(nEvtBadTiming[i])/double(nEvtTot[i]);
      BadSignalEvtPercentage=double(nEvtBadSignal[i])/double(nEvtTot[i]);
    }
    if(BadGainEvtPercentage>_qualpercent) {
      wasGainOK[i]=false;
      nChanBadGain++;
    }
    if(BadTimingEvtPercentage>_qualpercent){
      wasTimingOK[i]=false;
      nChanBadTiming++;
    }
    if(BadSignalEvtPercentage>_qualpercent){
      wasSignalOK[i]=false;
      nChanBadSignal++;
    }
  }

  double BadGainChanPercentage=double(nChanBadGain)/double(nCrys);
  double BadTimingChanPercentage=double(nChanBadTiming)/double(nCrys);
  double BadSignalChanPercentage=double(nChanBadSignal)/double(nCrys);
  
  if(BadGainChanPercentage>_qualpercent) isGainOK = false;
  if(BadTimingChanPercentage>_qualpercent) isTimingOK = false;
  if(BadSignalChanPercentage>_qualpercent) isSignalOK = false;


  if ( _debug == 1 ) cout << "-- debug EcalABAnalyzer -- endJob - isGainOK = "<< 
		       isGainOK << " isTimingOK = "<<isTimingOK<< endl;


  // Adjust channel numbers for EE 
  //===============================

  if( _ecalPart == "EE" ) {
    for(unsigned int ic=0;ic<colors.size();ic++){
      shapana[ic]->set_nch(nCrys);
    }
  }

  cout <<  "\n\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;
  if( _type == "LASER" ) 
    cout <<    "\t+=+     Analyzing LASER data: getting (alpha, beta)     +=+" << endl;
  else if( _type == "LED" ) 
    cout <<    "\t+=+      Analyzing LED data: getting (alpha, beta)      +=+" << endl;

  for(unsigned int ic=0;ic<colors.size();ic++){
    
    // Set presamples number 
    //======================
    shapana[ic]->set_presample(_presample);
    
    //  Get alpha and beta
    //=====================

    if(doesABInitTreeExist){
      TFile fAB(alphainitfile.c_str()); 
      TTree *ABT=0;
      if( ! fAB.IsZombie() ){
	ABT = (TTree*) fAB.Get("ABCol0"); 
      }
      shapana[ic]->computeShape(alphafile, ABT); 
      fAB.Close();   
    }else{
      shapana[ic]->computeShape(alphafile); 
    }
    
  }
  
  
  cout <<    "\t+=+       .................................... done     +=+" << endl;
  cout <<    "\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;
  
}
  




  
DEFINE_FWK_MODULE(EcalABAnalyzer);

