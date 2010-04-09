/* 
 *  \class EcalLightAnalyzer
 *
 *  $Date: 2009/06/02 12:55:19 $
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
#include <TH1D.h>

#include "CalibCalorimetry/EcalLaserAnalyzer/plugins/EcalLightAnalyzer.h"

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
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TPNFit.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMom.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TAPD.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TAPDPulse.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TPN.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TPNPulse.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TCalibData.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMem.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TSFit.h>  
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/PulseFitWithFunction.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/PulseFitWithShape.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/ME.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/MEGeom.h>


//========================================================================
EcalLightAnalyzer::EcalLightAnalyzer(const edm::ParameterSet& iConfig)
//========================================================================
  :
iEvent(0),

// Framework parameters with default values
 
_type    (        iConfig.getUntrackedParameter< std::string  >( "type",          "LASER" ) ), 
_typefit    (     iConfig.getUntrackedParameter< std::string  >( "typefit",          "AB" ) ), 

// COMMON:
_nsamples(        iConfig.getUntrackedParameter< unsigned int >( "nSamples",           10 ) ),
_presample(       iConfig.getUntrackedParameter< unsigned int >( "nPresamples",         3 ) ),
_firstsample(     iConfig.getUntrackedParameter< unsigned int >( "firstSample",         1 ) ),
_lastsample(      iConfig.getUntrackedParameter< unsigned int >( "lastSample",          2 ) ),
_nsamplesPN(      iConfig.getUntrackedParameter< unsigned int >( "nSamplesPN",         50 ) ), 
_presamplePN(     iConfig.getUntrackedParameter< unsigned int >( "nPresamplesPN",       6 ) ),
_firstsamplePN(   iConfig.getUntrackedParameter< unsigned int >( "firstSamplePN",       7 ) ),
_lastsamplePN(    iConfig.getUntrackedParameter< unsigned int >( "lastSamplePN",        8 ) ),
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
_niter(           iConfig.getUntrackedParameter< unsigned int >( "nIter",               5 ) ),
_noise(           iConfig.getUntrackedParameter< double       >( "noise",             1.2 ) ),
_ecalPart(        iConfig.getUntrackedParameter< std::string  >( "ecalPart",         "EB" ) ),
_fedid(           iConfig.getUntrackedParameter< int          >( "fedID",            -999 ) ),
_saveallevents(   iConfig.getUntrackedParameter< bool         >( "saveAllEvents",   false ) ),
_debug(           iConfig.getUntrackedParameter< int          >( "debug",              0  ) ),
_t0shapes(        iConfig.getUntrackedParameter< double       >( "t0shapes",             10. ) ), 

// LASER/LED AB:
_fitab(           iConfig.getUntrackedParameter< bool         >( "fitAB",           false ) ),

// LASER/LED SHAPE:

_nsamplesshapes(                                                                NSAMPSHAPES ),
_npresamplesshapes(                                                                      50 ), 

// COMMON:
nCrys(                                                                               NCRYSEB),
nPNPerMod(                                                                         NPNPERMOD),
nMod(                                                                                 NMODEE),
nSides(                                                                               NSIDES),
runType(-1), runNum(0), fedID(-1), dccID(-1), side(2), lightside(2), iZ(1),
phi(-1), eta(-1), event(0), color(-1),pn0(0), pn1(0), apdAmpl(0), apdAmplA(0), 
apdAmplB(0),apdTime(0),pnAmpl(0), pnAmplCor(0),
pnID(-1), moduleID(-1), channelIteratorEE(0),

// LASER/LED SHAPE:

isMatacqOK(false), 

// TEST PULSE:

nGainPN(                                                                       NGAINPN),
nGainAPD(                                                                     NGAINAPD)

//========================================================================

{

  // Initialization from cfg file

  resdir_                 = iConfig.getUntrackedParameter<std::string>("resDir");
  calibpath_              = iConfig.getUntrackedParameter<std::string>("calibPath", "/nfshome0/ecallaser/calibdata/140110"); 
  alphainitpath_              = iConfig.getUntrackedParameter<std::string>("alphaPath", "/nfshome0/ecallaser/calibdata/140110/alphabeta");

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

  if( _typefit != "AB" && _typefit != "SHAPE" ){
    cout << "Error: Wrong typefit "<<_typefit<<", should be AB or SHAPE"<< endl;
    cout << " Setting typefit to AB "<<endl;
    _typefit="AB";
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
    meanRawAmpl[j]=0.0;
    nEvtRawAmpl[j]=0;
  }

  for(int icc=0;icc<nColor;icc++){
    for(int iss=0;iss<NSIDES;iss++){      
      gotLasShapeForPN[icc][iss]=false;
      for(unsigned int j=0;j<NPN;j++){
	_corrPNEB[j][icc][iss]=-1.;
	for(unsigned int i=0;i<NMEMEE;i++){
	  _corrPNEE[j][i][icc][iss]=-1.;
	}
      }
    }
  }
  
  meanMeanRawAmpl=0.0;

  for(unsigned int j=0;j<nMod;j++){
    int ii= modules[j];
    firstChanMod[ii-1]=0;
    isFirstChanModFilled[ii-1]=0;
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
 
  PNPulse = new TPNPulse(_nsamplesPN, _presamplePN);

  // Object dealing with MEM numbering

  Mem = new TMem(_fedid);

  // Objects needed for npresample calculation

  Delta01=new TMom();
  Delta12=new TMom();

}

//========================================================================
EcalLightAnalyzer::~EcalLightAnalyzer(){
  //========================================================================


  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)

}



//========================================================================
void EcalLightAnalyzer::beginJob() {
//========================================================================

  
  // Create temporary files and trees to save adc samples
  //======================================================

  ADCfile=resdir_;
  ADCfile+="/APDSamples";
  ADCfile+=_type;
  ADCfile+=_typefit;
  if(_typefit!="SHAPE"){
    if(_fitab) ADCfile+="Fit";
    else ADCfile+="Fix";
  }
  ADCfile+=".root";

  APDfile=resdir_;
  APDfile+="/APDAmplitudes";  
  APDfile+=_type;
  APDfile+=_typefit;
  if(_typefit!="SHAPE"){
    if(_fitab) APDfile+="Fit";
    else APDfile+="Fix";
  }
  APDfile+=".root";

  ADCFile = new TFile(ADCfile.c_str(),"RECREATE");
  
  for (unsigned int i=0;i<nCrys;i++){
    
    stringstream name;
    name << "ADCTree" <<i+1;
    ADCTrees[i]= new TTree(name.str().c_str(),name.str().c_str());

    ADCTrees[i]->Branch( "ieta",        &eta,         "eta/I"         );
    ADCTrees[i]->Branch( "iphi",        &phi,         "phi/I"         );
    ADCTrees[i]->Branch( "side",        &side,        "side/I"        );
    ADCTrees[i]->Branch( "dccID",       &dccID,       "dccID/I"       );
    ADCTrees[i]->Branch( "towerID",     &towerID,     "towerID/I"     );
    ADCTrees[i]->Branch( "channelID",   &channelID,   "channelID/I"   );
    ADCTrees[i]->Branch( "event",       &event,       "event/I"       );
    ADCTrees[i]->Branch( "color",       &color,       "color/I"       );
    ADCTrees[i]->Branch( "adc",         &adc ,        "adc[10]/D"     );
    ADCTrees[i]->Branch( "adcGain",     &adcGain ,    "adcGain/I"     );
    ADCTrees[i]->Branch( "pn0",         &pn0 ,        "pn0/D"         );
    ADCTrees[i]->Branch( "pn1",         &pn1 ,        "pn1/D"         );
    ADCTrees[i]->Branch( "pn0cor",      &pn0cor ,     "pn0cor/D"      );
    ADCTrees[i]->Branch( "pn1cor",      &pn1cor ,     "pn1cor/D"      );
    ADCTrees[i]->Branch( "pnGain",      &pnGain,      "pnGain/I"      );
     
    ADCTrees[i]->SetBranchAddress( "ieta",        &eta         );  
    ADCTrees[i]->SetBranchAddress( "iphi",        &phi         ); 
    ADCTrees[i]->SetBranchAddress( "side",        &side        ); 
    ADCTrees[i]->SetBranchAddress( "dccID",       &dccID       ); 
    ADCTrees[i]->SetBranchAddress( "towerID",     &towerID     ); 
    ADCTrees[i]->SetBranchAddress( "channelID",   &channelID   ); 
    ADCTrees[i]->SetBranchAddress( "event",       &event       );   
    ADCTrees[i]->SetBranchAddress( "color",       &color       );   
    ADCTrees[i]->SetBranchAddress( "adc",         adc          );
    ADCTrees[i]->SetBranchAddress( "pn0",         &pn0         );
    ADCTrees[i]->SetBranchAddress( "pn1",         &pn1         ); 
    ADCTrees[i]->SetBranchAddress( "pn0cor",      &pn0cor      );
    ADCTrees[i]->SetBranchAddress( "pn1cor",      &pn1cor      ); 
    ADCTrees[i]->SetBranchAddress( "pnGain",      &pnGain      );    
 
  } 
 
  if ( _debug == 1 ) cout << "-- debug EcalLightAnalyzer --  beginJob - _type = "<<_type<<" _typefit = "<<_typefit<< endl;
  
  // Get Shapes Anyway:
  //=====================
  calibData= new TCalibData( _fedid, calibpath_ );


  
  // Good type events counter  

  typeEvents=0;
  
  if( _type != "LASER" && _type!= "LED"){
    cout<<" Wrong type"<< endl;
  }

  // APD results file:
  //===================
  stringstream nameapdfile;
  if(_typefit=="SHAPE") nameapdfile << resdir_ <<"/APDPN_"<<_type<<".root";    
  else if(_fitab) nameapdfile << resdir_ <<"/APDPN_"<<_type<<"_ABFit.root"; 
  else nameapdfile << resdir_ <<"/APDPN_"<<_type<<"_ABFix.root"; 
  
  resfile=nameapdfile.str();

  stringstream namefile3;
  if(_type=="LASER"){
    namefile3 << resdir_ <<"/MATACQ.root";      
    matfile=namefile3.str();
  }
}


//========================================================================
void EcalLightAnalyzer:: analyze( const edm::Event & e, const  edm::EventSetup& c){
//========================================================================

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
      if(_debug==1) cout<<" New color : " <<color<< endl;      
    }

    vector<int>::iterator iterside = find( sides.begin(), sides.end(), lightside );
    if( iterside==sides.end() ){
      sides.push_back( lightside );
    }

  }

  // Cut on fedID
  
  if(fedID!=_fedid && _fedid!=-999) return; 
  
  // Count good type events 

  typeEvents++;


  // ======================
  // Decode PN Information
  // ======================
  
  TPNFit * pnfit = new TPNFit();
  pnfit -> init(_nsamplesPN,_firstsamplePN,  _lastsamplePN);
  
  double chi2pn=0;
  double pnTime=0;
  unsigned int samplemax=0;
  double valmax=0.;
  double valped=0.;
  pnGain=0;
  
  map <int, vector<double> > allPNAmpl;
  map <int, vector<double> > allPNAmplCor;
  map <int, vector<double> > allPNGain;

  
  // Loop on PNs digis
  
  for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNDigi->begin();
	pnItr != PNDigi->end(); ++pnItr ) { 
    
    EcalPnDiodeDetId pnDetId = EcalPnDiodeDetId((*pnItr).id());
    
    if ( _debug == 2 ) cout <<"-- debug EcalLightAnalyzer -- analyze: in PNDigi - pnID=" <<
		     pnDetId.iPnId()<<", dccID="<< pnDetId.iDCCId()<< endl;

    // Skip MEM DCC without relevant data
  
    bool isMemRelevant=Mem->isMemRelevant(pnDetId.iDCCId());
    if(!isMemRelevant) continue;

    // Loop on PN samples

    for ( int samId=0; samId < (*pnItr).size() ; samId++ ) {   
      pn[samId]=(*pnItr).sample(samId).adc();  
      pnG[samId]=(*pnItr).sample(samId).gainId(); 
      if (samId==0) pnGain=pnG[samId];
      if (samId>0) pnGain=int(TMath::Max(pnG[samId],pnGain));     
    }
    
    // Calculate amplitude from pulse
    
    PNPulse->setPulse(pn);

    pnNoPed=PNPulse->getAdcWithoutPedestal();
    samplemax=PNPulse->getMaxSample();
    valmax=PNPulse->getMax();
    valped=PNPulse->getPedestal();


    // Don't fit for PN without signal
    if(valmax-valped< 20.0 ){
      allPNAmpl[pnDetId.iDCCId()].push_back(0.);
      allPNAmplCor[pnDetId.iDCCId()].push_back(0.);
      continue;
    }
    
    chi2pn = pnfit -> doFit(samplemax,&pnNoPed[0]); 
    
    if(chi2pn == 101 || chi2pn == 102 || chi2pn == 103){
      pnAmpl=0.; 
      pnTime=0.;
    }else{
      pnAmpl= pnfit -> getAmpl();
      pnTime= pnfit -> getTime();
    }
    if ( _debug == 3 ) cout<< "PN AMPL AFTER NO FIT:     "<<valmax-valped<<" "<<samplemax<< " "<< valped<< endl;
    
    pair<double,double> tauspn=calibData->tauPN( pnDetId.iPnId()-1 , pnDetId.iDCCId() );
    double qmax=calibData->qmaxPN(  pnDetId.iPnId()-1 , pnDetId.iDCCId() );

    if ( _debug == 3 ) cout<<"PN ID:"<<pnDetId.iPnId()-1 <<" MEM:" << pnDetId.iDCCId()<< " TAUS: (" <<tauspn.first<<","<<tauspn.second<<") QMAX:"<<qmax<< endl; 

    if ( _debug == 3 ) cout<< "PN AMPL AFTER FIRST FIT:  "<<pnAmpl<< " "<< pnTime<< " "<<chi2pn<<" "<< qmax<<" "<<endl;
    
    chi2pn = pnfit -> doFit2(&pnNoPed[0], tauspn.first, tauspn.second, 
			     pnAmpl, pnTime, qmax ); 

    
    // Protect against bad fit2 by comparing results:
    
    if(pnAmpl==0 || TMath::Abs(pnfit -> getAmpl2()-pnAmpl)/pnAmpl<0.01) {

      pnAmpl= pnfit -> getAmpl2();
      pnTime= pnfit -> getTime2();
      
    }else
      cout<< "PN AMPL DIFFERENT AFTER FIRST AND SECOND FIT: "<<pnAmpl<<" "<<pnfit->getAmpl2()<<endl;
    
    if ( _debug == 3 ) cout<< "PN AMPL AFTER SECOND FIT: "<<pnAmpl<< " "<< pnTime<<" "<<chi2pn<< " "<< endl;
    
    
    // Apply linearity correction

    pnAmplCor=calibData->getPNCorrected(pnAmpl,pnDetId.iPnId()-1,0, pnDetId.iDCCId()); // pngain==1 ==> indicegainmarc=0 pour le gain 16!!!
    
    
    // Fill PN ampl vector    
    allPNAmpl[pnDetId.iDCCId()].push_back(pnAmpl);
    allPNAmplCor[pnDetId.iDCCId()].push_back(pnAmplCor);
    if ( _debug == 2 ) cout <<"-- debug EcalLightAnalyzer -- analyze: in PNDigi - PNampl=" << 
			 pnAmpl<<", PNamplCor=" << 
			 pnAmplCor<<", PNgain="<< pnGain<<endl;  
  }
  
  // ===========================
  // Decode EBDigis Information
  // ===========================

  adcGain=0;

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
           
      int apdRefTT=MEEBGeom::apdRefTower(module);      
           
      std::pair<int,int> pnpair=MEEBGeom::pn(module, _fedid);
      unsigned int MyPn0=pnpair.first;
      unsigned int MyPn1=pnpair.second;
      
      int lmr=MEEBGeom::lmr( etaG,phiG ); 
      unsigned int channel=MEEBGeom::electronic_channel( etaL, phiL );

      assert( channel <= nCrys );
      
      side=MEEBGeom::side(etaG,phiG);

      if( _type == "LASER" && lightside!=side) continue;

      assert( module>=*min_element(modules.begin(),modules.end()) && module<=*max_element(modules.begin(),modules.end()) );
      
      eta = etaG;
      phi = phiG;
      channelID=5*(strip-1) + xtal-1; 
      towerID=tower;
  
      vector<int> apdRefChan=ME::apdRefChannels(module, lmr);      
      for (unsigned int iref=0;iref<nRefChan;iref++){
	if(channelID==apdRefChan[iref] && towerID==apdRefTT 
	   && apdRefMap[iref].count(module)==0){
	  apdRefMap[iref][module]=channel;
	}
      }
  
      if(isFirstChanModFilled[module-1]==0) {
	firstChanMod[module-1]=channel;
	isFirstChanModFilled[module-1]=1;
      } 
      if ( _debug == 2 ) cout << "-- debug EcalLightAnalyzer -- analyze: in EBDigi - towerID:"<< towerID<<
		       " channelID:" <<channelID<<" module:"<< module<<
		       " modules:"<<modules.size()<< endl;
      
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

      // Associate PN ampl
      //===================
 
      int mem0=Mem->Mem(lmr,0);
      int mem1=Mem->Mem(lmr,1);
      
      if(allPNAmpl[mem0].size()>MyPn0){
	pn0=allPNAmpl[mem0][MyPn0];
	pn0cor=allPNAmplCor[mem0][MyPn0];
	
      }else{
	pn0=0;
	pn0cor=0;
      }
      if(allPNAmpl[mem1].size()>MyPn1){
	pn1=allPNAmpl[mem1][MyPn1];
	pn1cor=allPNAmplCor[mem1][MyPn1];
      }else{
	pn1=0;
	pn1cor=0;
      
      }
      
    
      // Fill if Pulse is fine
      //=======================

      meanRawAmpl[channel]+=(APDPulse->getMax()-APDPulse->getPedestal());
      nEvtRawAmpl[channel]++;
      
      if( APDPulse->isPulseOK() ){
	
	ADCTrees[channel]->Fill(); 
	
	Delta01->addEntry(APDPulse->getDelta(0,1));
	Delta12->addEntry(APDPulse->getDelta(1,2));
	
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

      assert( module>=*min_element(modules.begin(),modules.end()) 
	      && module<=*max_element(modules.begin(),modules.end()) );
      
      if ( _debug == 2 ) cout << "-- debug EcalLightAnalyzer -- analyze: in EEDigi -  module:"<< module<<
	" modules:"<<modules.size()<<" side: "<< side<<endl;
      
      int lmr=MEEEGeom::lmr( iX, iY ,iZ); 
      int dee=MEEEGeom::dee(lmr);
      int apdRefTT=MEEEGeom::apdRefTower(lmr, module);  
      
      std::pair<int,int> pnpair=MEEEGeom::pn( dee, module ) ; 
      unsigned int MyPn0=pnpair.first;
      unsigned int MyPn1=pnpair.second;
      
      int hashedIndex=100000*etaG+phiG;
      if( channelMapEE.count(hashedIndex) == 0 ){
      	channelMapEE[hashedIndex]=channelIteratorEE;
      	channelIteratorEE++;
      }
      unsigned int channel=channelMapEE[hashedIndex];

      if( _type == "LASER" && lightside!=side) continue;
      
      
      eta = etaG;
      phi = phiG;
      channelID=ch;
      towerID=tower;
      
      vector<int> apdRefChan=ME::apdRefChannels(module, lmr);      
      for (unsigned int iref=0;iref<nRefChan;iref++){
	if(channelID==apdRefChan[iref] && towerID==apdRefTT 
	   && apdRefMap[iref].count(module)==0){
	  apdRefMap[iref][module]=channel;
	}
      }
      
      if(isFirstChanModFilled[module-1]==0) {
	firstChanMod[module-1]=channel;
	isFirstChanModFilled[module-1]=1;
	if ( _debug == 2 ) cout << "-- debug EcalLightAnalyzer -- analyze: filling first chan mod: " << module<<"  " << channel<<"  " <<eta<<"  " << phi<< endl;
      }
      
      assert ( channel < nCrys );
      
      if ( _debug == 2 ) cout << "-- debug EcalLightAnalyzer -- analyze: in EEDigi - towerID:"<< towerID<<
		       " channelID:" <<channelID<<" module:"<< module<<
	" modules:"<<modules.size()<<" side: "<< side<<endl;
      
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

      
      // Associate PN ampl
      //===================

      int mem0=Mem->Mem(lmr,0);
      int mem1=Mem->Mem(lmr,1);
      
      if ( _debug == 2 ) cout << "-- debug EcalLightAnalyzer -- analyze: in EEDigi - mems: "<< mem0<<" "<<mem1<<" "<< MyPn0<<" "<<MyPn1<<" "<< lmr<< endl;
      
      if ( _debug == 2 ) cout << "-- debug EcalLightAnalyzer -- analyze: in EEDigi - mems: "<< allPNAmpl[mem0].size()<<" "<<allPNAmpl[mem0].size()<<endl;
      if(allPNAmpl[mem0].size()>MyPn0){
	pn0=allPNAmpl[mem0][MyPn0];
	pn0cor=allPNAmplCor[mem0][MyPn0];
      }else{
	pn0=0; pn0cor=0;
      }
      if(allPNAmpl[mem1].size()>MyPn1){
	pn1=allPNAmpl[mem1][MyPn1];
	pn1cor=allPNAmplCor[mem1][MyPn1];
      }else{
	pn1=0;
	pn1cor=0;
      }

      // Fill if Pulse is fine
      //=======================
      
      meanRawAmpl[channel]+=(APDPulse->getMax()-APDPulse->getPedestal());
      nEvtRawAmpl[channel]++;

      if( APDPulse->isPulseOK()){
	if ( _debug == 2 ) cout << "-- debug EcalLightAnalyzer -- analyze: in EEDigi - Filling: "<<endl;
	ADCTrees[channel]->Fill(); 
	
	Delta01->addEntry(APDPulse->getDelta(0,1));
	Delta12->addEntry(APDPulse->getDelta(1,2));
	
      }
    }
  }
}

void EcalLightAnalyzer::endJob() {
  
  
  if ( _debug == 2 ) cout << "-- debug EcalLightAnalyzer -- entering endJob events="<<typeEvents<<endl;

  // Don't do anything if there is no events
  //=========================================

  if( typeEvents == 0 ){
    
    if(ADCFile) ADCFile->Close(); 
    
    stringstream del;
    del << "rm " <<ADCfile;
    system(del.str().c_str());


    cout << "\n\t***  No "<< _type<< " Events  ***"<< endl;
    
    return;
  }
  
  // Get Pulse Shapes
  //==================
  
  isMatacqOK=getMatacq();   
  
  if(!isMatacqOK){
    
    cout << "\n\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;
    cout <<   "\t+=+  LIGHT ANALYZER: WARNING! NO PULSE SHAPE    +=+" << endl;
    cout <<   "\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;
    if(ADCFile) ADCFile->Close(); 
    stringstream del;
    del << "rm " <<ADCfile;
    system(del.str().c_str());
    return;
  }
  
  // Get AB Shapes
  //==================
  if(_typefit == "AB" ){

    doesABTreeExist=getAB();
    
    if(!doesABTreeExist){
      
      cout << "\n\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;
      cout <<   "\t+=+     WARNING! NO AB FILE       +=+" << endl;
      cout <<   "\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;
      if(ADCFile) ADCFile->Close(); 
      
      stringstream del;
      del << "rm " <<ADCfile;
      system(del.str().c_str());
      return;      
    }  
  }
  
  if ( _debug == 1 ) cout << "-- debug EcalLightAnalyzer -- endJob - still Here"<< endl;
  
  // Adjust channel numbers for EE 
  //===============================

  if( _ecalPart == "EE" ) {
    nCrys=channelMapEE.size();
  }
  if ( _debug == 1 ) cout << "-- debug EcalLightAnalyzer -- endJob - nCrys = "<< nCrys << endl;
  
  // Set presamples number 
  //======================
  
  double delta01=Delta01->getMean();
  double delta12=Delta12->getMean();
  if(delta12>_presamplecut) {
    _presample=2;
    if(delta01>_presamplecut) _presample=1;
  }
  
  
  if ( _debug == 2 ){
  cout<< " PRESAMPLE CHECK: " <<delta01<<"  " <<delta12<<"  " <<_presample<< endl;
  cout<< " NOISE CHECK: " <<_noise<<" " <<_nsamples<<"  " <<_firstsample<<"  " <<_lastsample<<"  " <<_niter<<" " << _nsamplesshapes<< endl;
  }
  
  APDPulse->setPresamples(_presample);

  if ( _debug == 1 ) cout << "-- debug EcalLightAnalyzer -- endJob - _presample = "<< _presample << endl;


  // Set quality flags for gains, timing, signal
  //=============================================

  double BadGainEvtPercentage=0.0;
  double BadTimingEvtPercentage=0.0;
  double BadSignalEvtPercentage=0.0;
  
  int nChanBadGain=0;
  int nChanBadTiming=0;
  int nChanBadSignal=0;
  
  for (unsigned int i=0;i<nCrys;i++){
    
    if(nEvtRawAmpl[i]!=0) meanRawAmpl[i]=meanRawAmpl[i]/double(nEvtRawAmpl[i]);
    else meanRawAmpl[i]=0.0;

    meanMeanRawAmpl+=meanRawAmpl[i]/double(nCrys);
    

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


  if ( _debug == 1 ) cout << "-- debug EcalLightAnalyzer -- endJob - isGainOK = "<< 
		       isGainOK << " isTimingOK = "<<isTimingOK<< endl;

 
  // Analyze adc samples to get amplitudes
  //=======================================
  
  cout <<  "\n\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;
  if( _type == "LASER" ) 
    cout <<    "\t+=+     Analyzing LASER data: getting APD, PN, APD/PN, PN/PN    +=+" << endl; 
  else if( _type == "LED" ) 
    cout <<    "\t+=+      Analyzing LED data: getting APD, PN, APD/PN, PN/PN     +=+" << endl;
  
  if( !isGainOK )
    cout <<    "\t+=+ ............................ WARNING! APD GAIN WAS NOT 1    +=+" << endl;
  if( !isTimingOK )
    cout <<    "\t+=+ ............................ WARNING! TIMING WAS BAD        +=+" << endl;  
  if( !isSignalOK )
    cout <<    "\t+=+ ............................ WARNING! LIGHT SIGNAL WAS BAD  +=+" << endl;
  cout <<    "\t+=+ ............................ MEAN RAW SIG: "<< meanMeanRawAmpl << endl;
  
  APDFile = new TFile(APDfile.c_str(),"RECREATE");
  
  int flag;
  
  for (unsigned int i=0;i<nCrys;i++){
    
    stringstream name;
    name << "APDTree" <<i+1;
    
    APDTrees[i]= new TTree(name.str().c_str(),name.str().c_str());
    
    //List of branches
    
    APDTrees[i]->Branch( "event",     &event,     "event/I"     );
    APDTrees[i]->Branch( "color",     &color,     "color/I"     );
    APDTrees[i]->Branch( "iphi",      &phi,       "iphi/I"      );
    APDTrees[i]->Branch( "ieta",      &eta,       "ieta/I"      );
    APDTrees[i]->Branch( "side",      &side,      "side/I"      );
    APDTrees[i]->Branch( "dccID",     &dccID,     "dccID/I"     );
    APDTrees[i]->Branch( "towerID",   &towerID,   "towerID/I"   );
    APDTrees[i]->Branch( "channelID", &channelID, "channelID/I" );
    APDTrees[i]->Branch( "apdAmpl",   &apdAmpl,   "apdAmpl/D"   );
    APDTrees[i]->Branch( "apdTime",   &apdTime,   "apdTime/D"   );
    if(_saveallevents) APDTrees[i]->Branch( "adc", &adc ,"adc[10]/D" );
    APDTrees[i]->Branch( "flag",      &flag,      "flag/I"      ); 
    APDTrees[i]->Branch( "pn0",       &pn0,       "pn0/D"       );
    APDTrees[i]->Branch( "pn1",       &pn1,       "pn1/D"       ); 
    APDTrees[i]->Branch( "pn0cor",    &pn0cor,    "pn0cor/D"    );
    APDTrees[i]->Branch( "pn1cor",    &pn1cor,    "pn1cor/D"    ); 
    
    APDTrees[i]->SetBranchAddress( "event",     &event     );
    APDTrees[i]->SetBranchAddress( "color",     &color     ); 
    APDTrees[i]->SetBranchAddress( "iphi",      &phi      );
    APDTrees[i]->SetBranchAddress( "ieta",      &eta      );
    APDTrees[i]->SetBranchAddress( "side",      &side      );
    APDTrees[i]->SetBranchAddress( "dccID",     &dccID     );
    APDTrees[i]->SetBranchAddress( "towerID",   &towerID   );
    APDTrees[i]->SetBranchAddress( "channelID", &channelID ); 
    APDTrees[i]->SetBranchAddress( "apdAmpl",   &apdAmpl   );
    APDTrees[i]->SetBranchAddress( "apdTime",   &apdTime   );
    if(_saveallevents)APDTrees[i]->SetBranchAddress( "adc", adc );
    APDTrees[i]->SetBranchAddress( "flag",      &flag      ); 
    APDTrees[i]->SetBranchAddress( "pn0",       &pn0       );
    APDTrees[i]->SetBranchAddress( "pn1",       &pn1       ); 
    APDTrees[i]->SetBranchAddress( "pn0cor",    &pn0cor    );
    APDTrees[i]->SetBranchAddress( "pn1cor",    &pn1cor    ); 
  
  }


  for (unsigned int iref=0;iref<nRefChan;iref++){
    for (unsigned int imod=0;imod<nMod;imod++){
      
      int jmod=modules[imod];
      
      stringstream nameref;
      nameref << "refAPDTree" <<imod<<"_"<<iref;
      
      refAPDTrees[iref][jmod]= new TTree(nameref.str().c_str(),nameref.str().c_str());
      
      refAPDTrees[iref][jmod]->Branch( "eventref",     &eventref,     "eventref/I"     );
      refAPDTrees[iref][jmod]->Branch( "colorref",     &colorref,     "colorref/I"     );
      if(iref==0) refAPDTrees[iref][jmod]->Branch( "apdAmplA",   &apdAmplA,   "apdAmplA/D"   );
      if(iref==1) refAPDTrees[iref][jmod]->Branch( "apdAmplB",   &apdAmplB,   "apdAmplB/D"   );
      
      refAPDTrees[iref][jmod]->SetBranchAddress( "eventref",     &eventref     );
      refAPDTrees[iref][jmod]->SetBranchAddress( "colorref",     &colorref     ); 
      if(iref==0)refAPDTrees[iref][jmod]->SetBranchAddress( "apdAmplA",   &apdAmplA   );
      if(iref==1)refAPDTrees[iref][jmod]->SetBranchAddress( "apdAmplB",   &apdAmplB   );
      
    } 
  }
 
  assert( colors.size()<= nColor );
  unsigned int nCol=colors.size();

  // Declare PN stuff
  //===================
  
  for (unsigned int iM=0;iM<nMod;iM++){
    
    unsigned int iMod=modules[iM]-1;

    for (unsigned int ich=0;ich<nPNPerMod;ich++){
      for (unsigned int icol=0;icol<nCol;icol++){
	PNFirstAnal[iMod][ich][icol]=new TPN(ich);
	PNAnal[iMod][ich][icol]=new TPN(ich);
      }

    }
  }
  

  // Declare function for APD ampl fit
  //===================================

  PulseFits *psfit;

  if( _typefit == "AB" ) psfit = new PulseFitWithFunction(_ecalPart);
  else if( _typefit == "SHAPE" ) psfit = new PulseFitWithShape();
  else psfit=0;

  double chi2;

  for (unsigned int iCry=0;iCry<nCrys;iCry++){

    bool pulseFitInit[2];
    
    for (unsigned int icol=0;icol<nCol;icol++){

      // Declare APD stuff
      //===================
      
      APDFirstAnal[iCry][icol] = new TAPD();
      isThereDataADC[iCry][icol]=1;      
      stringstream cut;
      cut <<"color=="<<colors[icol];
      if(ADCTrees[iCry]->GetEntries(cut.str().c_str())<10) isThereDataADC[iCry][icol]=0;  
      
      pulseFitInit[icol]=false;
    }

    double alpha, beta;
    
    // Loop on events
    //================
    
    Long64_t nbytes = 0, nb = 0;
    for (Long64_t jentry=0; jentry< ADCTrees[iCry]->GetEntriesFast();jentry++){
      nb = ADCTrees[iCry]->GetEntry(jentry);   nbytes += nb; 	
      
      // Get back color
      
      unsigned int iCol=0;
      for(unsigned int i=0;i<nCol;i++){
	if(color==colors[i]) {
	  iCol=i;
	  i=colors.size();
	}
      }
      
      unsigned int iMod;
      if(_ecalPart=="EB") iMod=MEEBGeom::lmmod(eta, phi)-1;
      else {
	int iX = (eta-1)/5+1;
	int iY = (phi-1)/5+1;
	iMod=MEEEGeom::lmmod(iX,iY);
	if( iMod>=18 && side==1 ) iMod+=2;
	iMod--;
	// FIXME test this	
      }


      // Initialize Pulse Fit With Shape
      //================================
      
      if( _typefit == "SHAPE" && pulseFitInit[iCol]==false ){
	
	//cout<<"Initializing shape for crystal:"<< iCry<<" color:"<< iCol<< endl;
	
	bool convolutionOK=getConvolution( eta, phi, iCol ); 
	
	if( convolutionOK ){
	  
	  ((PulseFitWithShape*) psfit) -> init(_nsamples,_firstsample,_lastsample,_niter, _nsamplesshapes,_presample, shapesAPD[iCol][side], _noise );
	  pulseFitInit[iCol]=true;	  
	  
	}else{
	  cout<< "Convolution not OK => return "<< endl;

	  if(ADCFile) ADCFile->Close(); 
	  stringstream del;
	  del << "rm " <<ADCfile;
	  system(del.str().c_str());

	  if(APDFile) APDFile->Close(); 	  
	  stringstream del2;
	  del2 << "rm " <<APDfile;
	  system(del2.str().c_str());

	  return;
	  
	} 
      }
      
      // Initialize Alpha Beta Pulse Fit
      //=================================
      
      if( _typefit == "AB" && pulseFitInit[iCol]==false ){
	
	pair<double, double> abvals=calibData->AB( eta, phi, color );
	alpha=abvals.first;
	beta=abvals.second;
	
	if( _debug == 2 ) cout<<" CHECK AB "<< iCry<< " "<<eta<<" "<<phi<<" "<< color<<" "<< alpha<<" "<<beta<< endl;
	
	((PulseFitWithFunction*) psfit) -> init(_nsamples,_firstsample,_lastsample,_niter,  alpha,  beta);

	pulseFitInit[iCol]=true;
      }
      
      // Amplitude calculation
      //========================

      APDPulse->setPulse(adc);
      adcNoPed=APDPulse->getAdcWithoutPedestal();
      
      apdAmpl=0;
      apdAmplA=0;
      apdAmplB=0;
      apdTime=0;
      
      if( APDPulse->isPulseOK()){
	
	chi2 = psfit -> doFit(&adcNoPed[0]);
	apdTime = psfit -> getTime();
	apdAmpl = psfit -> getAmpl();
	flag=1;
	
	if(_debug==2 && jentry==0){
	  cout.precision(9);
	  cout << "AMPL: "<< iCry<<" "<<side<<"  " <<apdAmpl<<" "<<APDPulse->getPedestal()<<" "<<(adc[0]+adc[1]+adc[2])/3.0 << " "<< pn0<<" "<<pn1<< endl;	  
	  cout.precision(6);
	}
	
	if( chi2 < 0. || chi2 == 102 || chi2 == 101  ) {	  
	  apdAmpl=0;	  
	  apdTime=0;
	  flag=0;
	}
	
      }else {
	chi2=0.0;
	apdAmpl=0;	  
	apdTime=0;
	flag=0;
      }
      
      if ( _debug == 2 ) cout <<"-- debug EcalLightAnalyzer -- endJobLight - apdAmpl="<<apdAmpl<< 
			   ", apdTime="<< apdTime<<" "<< chi2<<" " << flag<< endl; 

      if ( _debug == 2 ) cout <<"-- debug EcalLightAnalyzer -- endJobLight - pn0="<<pn0<<", pn1="<<pn1<< endl;
      
      // Fill PN stuff
      //===============
      
      if( firstChanMod[iMod]==iCry && isThereDataADC[iCry][iCol]==1 ){ 
	for (unsigned int ichan=0;ichan<nPNPerMod;ichan++){
	  PNFirstAnal[iMod][ichan][iCol]->addEntry(pn0,pn1);

	  if ( _debug == 2 ) cout <<"-- debug EcalLightAnalyzer -- endJobLight - Adding to iMod:"<<iMod<<"  " <<ichan<<"  " <<iCol<< endl;
	}
      }
    
      // Fill APD stuff 
      //================

      if( APDPulse->isPulseOK() ){
	APDFirstAnal[iCry][iCol]->addEntry(apdAmpl, pn0, pn1, apdTime, 0., 0., pn0cor, pn1cor);      
	APDTrees[iCry]->Fill();    
	
	// Fill reference trees
	//=====================
	
	if( apdRefMap[0][iMod+1]==iCry || apdRefMap[1][iMod+1]==iCry ) {
	  
	  apdAmplA=0.0;
	  apdAmplB=0.0;      
	  eventref=event;
	  colorref=color;
	  
	  for(unsigned int ir=0;ir<nRefChan;ir++){
	    
	    if(apdRefMap[ir][iMod+1]==iCry) {
	      if (ir==0) apdAmplA=apdAmpl;
	      else if(ir==1) apdAmplB=apdAmpl;
	      refAPDTrees[ir][iMod+1]->Fill(); 
	    }	
	  }
	}
      }
    }
  }

  delete psfit;
  
  ADCFile->Close(); 
  
  // Remove temporary file
  //=======================
  stringstream del;
  del << "rm " <<ADCfile;
  system(del.str().c_str());
  
  
  // Create output trees
  //=====================

  resFile = new TFile(resfile.c_str(),"RECREATE");
  
  for (unsigned int iColor=0;iColor<nCol;iColor++){
    
    stringstream nametree;
    nametree <<"APDCol"<<colors[iColor];
    stringstream nametree2;
    nametree2 <<"PNCol"<<colors[iColor];
    
    resTrees[iColor]= new TTree(nametree.str().c_str(),nametree.str().c_str());
    resPNTrees[iColor]= new TTree(nametree2.str().c_str(),nametree2.str().c_str());
    
    resTrees[iColor]->Branch( "iphi",        &phi,         "iphi/I"           );
    resTrees[iColor]->Branch( "ieta",        &eta,         "ieta/I"           );
    resTrees[iColor]->Branch( "side",        &side,        "side/I"           );
    resTrees[iColor]->Branch( "dccID",       &dccID,       "dccID/I"          );
    resTrees[iColor]->Branch( "moduleID",    &moduleID,    "moduleID/I"       );
    resTrees[iColor]->Branch( "towerID",     &towerID,     "towerID/I"        );
    resTrees[iColor]->Branch( "channelID",   &channelID,   "channelID/I"      );
    resTrees[iColor]->Branch( "APD",         &APD,         "APD[6]/D"         );
    resTrees[iColor]->Branch( "Time",        &Time,        "Time[6]/D"        );
    resTrees[iColor]->Branch( "APDoPN",      &APDoPN,      "APDoPN[6]/D"      );
    resTrees[iColor]->Branch( "APDoPNA",     &APDoPNA,     "APDoPNA[6]/D"     );
    resTrees[iColor]->Branch( "APDoPNB",     &APDoPNB,     "APDoPNB[6]/D"     );
    resTrees[iColor]->Branch( "APDoPNCor",   &APDoPNCor,   "APDoPNCor[6]/D"   );
    resTrees[iColor]->Branch( "APDoPNACor",  &APDoPNACor,  "APDoPNACor[6]/D"  );
    resTrees[iColor]->Branch( "APDoPNBCor",  &APDoPNBCor,  "APDoPNBCor[6]/D"  );
    resTrees[iColor]->Branch( "APDoAPD",     &APDoAPD,     "APDoAPD[6]/D"     );
    resTrees[iColor]->Branch( "APDoAPDA",    &APDoAPDA,    "APDoAPDA[6]/D"    );
    resTrees[iColor]->Branch( "APDoAPDB",    &APDoAPDB,    "APDoAPDB[6]/D"    );
    resTrees[iColor]->Branch( "shapeCorAPD", &shapeCorAPD, "shapeCorAPD/D"    ); 
    resTrees[iColor]->Branch( "flag",        &flag,        "flag/I"           ); 

    resPNTrees[iColor]->Branch( "side",      &side,      "side/I"         );
    resPNTrees[iColor]->Branch( "moduleID",  &moduleID,  "moduleID/I"     );
    resPNTrees[iColor]->Branch( "pnID",      &pnID,      "pnID/I"         );
    resPNTrees[iColor]->Branch( "PN",        &PN,        "PN[6]/D"        ); 
    resPNTrees[iColor]->Branch( "PNoPN",     &PNoPN,     "PNoPN[6]/D"     );
    resPNTrees[iColor]->Branch( "PNoPNA",    &PNoPNA,    "PNoPNA[6]/D"    );
    resPNTrees[iColor]->Branch( "PNoPNB",    &PNoPNB,    "PNoPNB[6]/D"    );
    resPNTrees[iColor]->Branch( "shapeCorPN", &shapeCorPN, "shapeCorPN/D" ); 

    resTrees[iColor]->SetBranchAddress( "iphi",        &phi       );
    resTrees[iColor]->SetBranchAddress( "ieta",        &eta       );
    resTrees[iColor]->SetBranchAddress( "side",        &side       );
    resTrees[iColor]->SetBranchAddress( "dccID",       &dccID      );
    resTrees[iColor]->SetBranchAddress( "moduleID",    &moduleID   );
    resTrees[iColor]->SetBranchAddress( "towerID",     &towerID    );
    resTrees[iColor]->SetBranchAddress( "channelID",   &channelID  );
    resTrees[iColor]->SetBranchAddress( "APD",         APD         ); 
    resTrees[iColor]->SetBranchAddress( "Time",        Time        );   
    resTrees[iColor]->SetBranchAddress( "APDoPN",      APDoPN      ); 
    resTrees[iColor]->SetBranchAddress( "APDoPNA",     APDoPNA     );
    resTrees[iColor]->SetBranchAddress( "APDoPNB",     APDoPNB     );   
    resTrees[iColor]->SetBranchAddress( "APDoPNCor",   APDoPNCor   ); 
    resTrees[iColor]->SetBranchAddress( "APDoPNACor",  APDoPNACor  );
    resTrees[iColor]->SetBranchAddress( "APDoPNBCor",  APDoPNBCor  );
    resTrees[iColor]->SetBranchAddress( "APDoAPD",     APDoAPD     );
    resTrees[iColor]->SetBranchAddress( "APDoAPDA",    APDoAPDA    );
    resTrees[iColor]->SetBranchAddress( "APDoAPDB",    APDoAPDB    );
    resTrees[iColor]->SetBranchAddress( "shapeCorAPD",   &shapeCorAPD );
    resTrees[iColor]->SetBranchAddress( "flag",        &flag       ); 
   
    resPNTrees[iColor]->SetBranchAddress( "side",      &side     );
    resPNTrees[iColor]->SetBranchAddress( "moduleID",  &moduleID );
    resPNTrees[iColor]->SetBranchAddress( "pnID",      &pnID     );
    resPNTrees[iColor]->SetBranchAddress( "PN",        PN        );
    resPNTrees[iColor]->SetBranchAddress( "PNoPN",     PNoPN     );
    resPNTrees[iColor]->SetBranchAddress( "PNoPNA",    PNoPNA    );
    resPNTrees[iColor]->SetBranchAddress( "PNoPNB",    PNoPNB    );
    resPNTrees[iColor]->SetBranchAddress( "shapeCorPN", &shapeCorPN );
  
  }
    


  // Set Cuts for PN stuff
  //=======================

  for (unsigned int iM=0;iM<nMod;iM++){

    unsigned int iMod=modules[iM]-1;
    
    for (unsigned int ich=0;ich<nPNPerMod;ich++){
      for (unsigned int icol=0;icol<nCol;icol++){
	PNAnal[iMod][ich][icol]->setPNCut(PNFirstAnal[iMod][ich][icol]->getPN().at(0),
					  PNFirstAnal[iMod][ich][icol]->getPN().at(1));
      }
    }
  }
  
  // Build ref trees indexes
  //========================
  for(unsigned int imod=0;imod<nMod;imod++){
    int jmod=modules[imod];
    if( refAPDTrees[0][jmod]->GetEntries()!=0 && refAPDTrees[1][jmod]->GetEntries()!=0 ){
      refAPDTrees[0][jmod]->BuildIndex("eventref");
      refAPDTrees[1][jmod]->BuildIndex("eventref");
    }
  }
  
  
  // Final loop on crystals
  //=======================

  for (unsigned int iCry=0;iCry<nCrys;iCry++){ 
    
    // Set cuts on APD stuff
    //=======================
    
    for(unsigned int iCol=0;iCol<nCol;iCol++){   
      
      std::vector<double> lowcut;
      std::vector<double> highcut;
      double cutMin;
      double cutMax;

      cutMin=APDFirstAnal[iCry][iCol]->getAPD().at(0)-2.0*APDFirstAnal[iCry][iCol]->getAPD().at(1);
      if(cutMin<0) cutMin=0;
      cutMax=APDFirstAnal[iCry][iCol]->getAPD().at(0)+2.0*APDFirstAnal[iCry][iCol]->getAPD().at(1);
      
      lowcut.push_back(cutMin);
      highcut.push_back(cutMax);
      
      cutMin=APDFirstAnal[iCry][iCol]->getTime().at(0)-2.0*APDFirstAnal[iCry][iCol]->getTime().at(1);
      cutMax=APDFirstAnal[iCry][iCol]->getTime().at(0)+2.0*APDFirstAnal[iCry][iCol]->getTime().at(1); 
      lowcut.push_back(cutMin);
      highcut.push_back(cutMax);
      
      APDAnal[iCry][iCol]=new TAPD();
      APDAnal[iCry][iCol]->setAPDCut(APDFirstAnal[iCry][iCol]->getAPD().at(0),APDFirstAnal[iCry][iCol]->getAPD().at(1));
      APDAnal[iCry][iCol]->setAPDoPNCut(APDFirstAnal[iCry][iCol]->getAPDoPN().at(0),APDFirstAnal[iCry][iCol]->getAPDoPN().at(1));
      APDAnal[iCry][iCol]->setAPDoPN0Cut(APDFirstAnal[iCry][iCol]->getAPDoPN0().at(0),APDFirstAnal[iCry][iCol]->getAPDoPN0().at(1));
      APDAnal[iCry][iCol]->setAPDoPN1Cut(APDFirstAnal[iCry][iCol]->getAPDoPN1().at(0),APDFirstAnal[iCry][iCol]->getAPDoPN1().at(1));
      APDAnal[iCry][iCol]->setTimeCut(APDFirstAnal[iCry][iCol]->getTime().at(0),APDFirstAnal[iCry][iCol]->getTime().at(1));
      
      APDAnal[iCry][iCol]->setAPDoPNCorCut(APDFirstAnal[iCry][iCol]->getAPDoPNCor().at(0),APDFirstAnal[iCry][iCol]->getAPDoPNCor().at(1));
      APDAnal[iCry][iCol]->setAPDoPN0CorCut(APDFirstAnal[iCry][iCol]->getAPDoPN0Cor().at(0),APDFirstAnal[iCry][iCol]->getAPDoPN0Cor().at(1));
      APDAnal[iCry][iCol]->setAPDoPN1CorCut(APDFirstAnal[iCry][iCol]->getAPDoPN1Cor().at(0),APDFirstAnal[iCry][iCol]->getAPDoPN1Cor().at(1));

    }
    
    // Final loop on events
    //=======================

    Long64_t nbytes = 0, nb = 0;
    for (Long64_t jentry=0; jentry< APDTrees[iCry]->GetEntriesFast();jentry++) { 
      nb = APDTrees[iCry]->GetEntry(jentry);   nbytes += nb; 
      
      unsigned int iMod;
      if(_ecalPart=="EB") iMod=MEEBGeom::lmmod(eta, phi)-1;
      else{	
	int iX = (eta-1)/5+1;
	int iY = (phi-1)/5+1;
	iMod=MEEEGeom::lmmod(iX,iY);
	if( iMod>=18 && side==1 ) iMod+=2;
	iMod--;
      }

      // Get back color
      //================
      
      unsigned int iCol=0;
      for(unsigned int i=0;i<nCol;i++){
	if(color==colors[i]) {
	  iCol=i;
	  i=colors.size();
	}
      }

      // Get back shape correction for first event
      //============================================
      if( jentry==0 ){
	shapeCorAPD=1.0;
	getShapeCor(eta, phi, iCol, shapeCorAPD );
      }
      
      // Fill PN stuff
      //===============
      
      if( firstChanMod[iMod]==iCry && isThereDataADC[iCry][iCol]==1 ){ 
	for (unsigned int ichan=0;ichan<nPNPerMod;ichan++){
	  PNAnal[iMod][ichan][iCol]->addEntry(pn0,pn1);
	  if ( _debug == 2 ) cout <<"-- debug EcalLightAnalyzer -- endJobLight - Adding2 to iMod:"<<iMod<<"  " <<ichan<<"  " <<iCol<<" " <<pn0<<"  " <<pn1<< endl;
	}
      }
      
      // Get ref amplitudes
      //===================

      if ( _debug == 2 ) cout <<"-- debug EcalLightAnalyzer -- endJobLight - Last Loop event:"<<event<<" apdAmpl:"<< apdAmpl<< endl;
      apdAmplA = 0.0;
      apdAmplB = 0.0;
      
      for (unsigned int iRef=0;iRef<nRefChan;iRef++){
	refAPDTrees[iRef][iMod+1]->GetEntryWithIndex(event); 
      }
      
      if ( _debug == 2 ) cout <<"-- debug EcalLightAnalyzer -- endJobLight - Last Loop apdAmplA:"<<apdAmplA<< " apdAmplB:"<< apdAmplB<<", event:"<< event<<", eventref:"<< eventref<< endl;
      
      
      // Fill APD stuff
      //===============
      
      APDAnal[iCry][iCol]->addEntry( apdAmpl, pn0, pn1, apdTime, apdAmplA,
				     apdAmplB, pn0cor, pn1cor ); 
      
  
      // Get Final results at last entry:
      //====================================
      
      if( jentry==APDTrees[iCry]->GetEntriesFast()-1){
	moduleID=iMod+1;  
	if( moduleID>=20 ) moduleID-=2; // Trick to fix endcap specificity   
	
	// Get final results for APD
	//===========================
	
	for(unsigned int iColor=0;iColor<nCol;iColor++){
	  
	  std::vector<double> apdvec = APDAnal[iCry][iColor]->getAPD();
	  std::vector<double> apdpnvec = APDAnal[iCry][iColor]->getAPDoPN();
	  std::vector<double> apdpn0vec = APDAnal[iCry][iColor]->getAPDoPN0();
	  std::vector<double> apdpn1vec = APDAnal[iCry][iColor]->getAPDoPN1();
	  std::vector<double> apdpncorvec = APDAnal[iCry][iColor]->getAPDoPNCor();
	  std::vector<double> apdpn0corvec = APDAnal[iCry][iColor]->getAPDoPN0Cor();
	  std::vector<double> apdpn1corvec = APDAnal[iCry][iColor]->getAPDoPN1Cor();
	  std::vector<double> timevec = APDAnal[iCry][iColor]->getTime();
	  std::vector<double> apdapdvec = APDAnal[iCry][iColor]->getAPDoAPD();
	  std::vector<double> apdapd0vec = APDAnal[iCry][iColor]->getAPDoAPD0();
	  std::vector<double> apdapd1vec = APDAnal[iCry][iColor]->getAPDoAPD1();
	  
	  for(unsigned int i=0;i<apdpnvec.size();i++){
	    
	    APD[i]=apdvec[i];
	    APDoPN[i]=apdpnvec[i];
	    APDoPNA[i]=apdpn0vec[i];
	    APDoPNB[i]=apdpn1vec[i];
	    APDoPNCor[i]=apdpncorvec[i];
	    APDoPNACor[i]=apdpn0corvec[i];
	    APDoPNBCor[i]=apdpn1corvec[i];
	    APDoAPD[i]=apdapdvec[i];
	    APDoAPDA[i]=apdapd0vec[i];
	    APDoAPDB[i]=apdapd1vec[i];
	    Time[i]=timevec[i];
	    
	  } 

	  // Fill APD results trees
	  //========================	  
	  
	  if( !wasGainOK[iCry] || !wasTimingOK[iCry] || isThereDataADC[iCry][iColor]==0 ){
	    flag=0;
	  }else flag=1;
	  resTrees[iColor]->Fill();   
	}
     
      }
    
    }
  }

  // Get final results for PN 
  //==========================
  
  for (unsigned int iM=0;iM<nMod;iM++){
    unsigned int iMod=modules[iM]-1;

    for (unsigned int ch=0;ch<nPNPerMod;ch++){

      pnID=ch;
      moduleID=iMod+1;
      
      if(_ecalPart=="EB") side = MEEBGeom::side(moduleID);
      else if(moduleID<20) side = 0;
      else side = 1;

      
      int lmr=ME::lmr(_fedid, side );
      pair<int,int> mempn=ME::pn(lmr,moduleID,(ME::PN) pnID);
      
      if( moduleID>=20 ) moduleID-=2;  // Trick to fix endcap specificity
      
      for(unsigned int iColor=0;iColor<nCol;iColor++){
	
	std::vector<double> pnvec = PNAnal[iMod][ch][iColor]->getPN();
	std::vector<double> pnopnvec = PNAnal[iMod][ch][iColor]->getPNoPN();
	std::vector<double> pnopn0vec = PNAnal[iMod][ch][iColor]->getPNoPN0();
	std::vector<double> pnopn1vec = PNAnal[iMod][ch][iColor]->getPNoPN1();
	
	for(unsigned int i=0;i<pnvec.size();i++){
	  
	  PN[i]=pnvec[i];
	  PNoPN[i]=pnopnvec[i];
	  PNoPNA[i]=pnopn0vec[i];
	  PNoPNB[i]=pnopn1vec[i];
	}

	
	if(_debug==2) cout<<" getConvolutionPN " <<mempn.second<<"  " <<mempn.first<<"  " <<iColor<<"  " <<side<< endl;
	shapeCorPN=getConvolutionPN( mempn.second, mempn.first, iColor, side );
	

	// Fill PN results trees
	//========================

	resPNTrees[iColor]->Fill();
      }
    }
  }
 
  
  // Save results 
  //===============

  for (unsigned int i=0;i<nCol;i++){
    resTrees[i]->Write();
    resPNTrees[i]->Write();
  }  
  
  resFile->Close(); 
   
  // Remove temporary files
  //========================
  if(!_saveallevents){
    
    APDFile->Close();
    stringstream del2;
    del2 << "rm " <<APDfile;
    system(del2.str().c_str());
    
  }else {
    
    APDFile->cd();
    APDTrees[0]->Write();
    
    APDFile->Close();
    resFile->cd();
  }
  delete calibData;

  cout <<    "\t+=+    .................................................. done  +=+" << endl;
  cout <<    "\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;
}

//========================================================================
bool EcalLightAnalyzer::getMatacq() {
//========================================================================
  

  // Get Pulse From Matacq Analysis:
  //================================
  
  bool ok=false;
  
  int doesMatFileExist=0;
  int doesMatShapeExist=0;

  
  FILE *test2;   
  test2 = fopen(matfile.c_str(),"r");
  if (test2){
    doesMatFileExist=1; 
    fclose(test2);
  }
  
  if( _type == "LED" ){

    double tau1=0.;
    double tau2=12.;
    double t0=90.;
    int nbin=250;
    double shapeval[250];
    
    for (int icol=0;icol<colors.size();icol++){
      for (int iside=0;iside<sides.size();iside++){
	stringstream hname;
	hname<<"pulseShape"<<icol<<"_" <<iside; 
	
	pulseShape[icol][iside] = new TH1D(hname.str().c_str(),hname.str().c_str(),nbin,0.0,double(nbin));

	for(int is=0; is<t0; is++) shapeval[is]=0.;
	for(int is=t0; is<tau1+t0; is++) shapeval[is]=1.;
	for(int is=tau1+t0; is<250; is++) shapeval[is]=exp(-((double)is-tau1-t0)/tau2);
	
	double nphot=0.;	
	for(int is=0; is<nbin; is++){
	  pulseShape[icol][iside]->SetBinContent(is+1,shapeval[is]);
	  nphot+=shapeval[is];	  
	}
	if(nphot>0) pulseShape[icol][iside]->Scale(1.0/nphot);
      }
    }
    doesMatFileExist=1;
    doesMatShapeExist=1;
    
  }else{
    
    TFile *matShapeFile;
    if (doesMatFileExist==1){
      matShapeFile = new TFile(matfile.c_str());
      
      for(unsigned int icol=0;icol<colors.size();icol++){
	for(unsigned int iside=0;iside<sides.size();iside++){
	  
	  stringstream nameshape;
	  stringstream nameshape2;
	  
	  nameshape<<"LaserCol"<<colors[icol]<<"Side"<<iside;
	  laserShape[icol][iside]= (TH1D*) matShapeFile->Get(nameshape.str().c_str());
	  
	  nameshape2<<"LaserMarcCol"<<colors[icol]<<"Side"<<iside;
	  pulseShape[icol][iside]= (TH1D*) matShapeFile->Get(nameshape2.str().c_str());
	  
	  if(laserShape[icol][iside]){
	    
	    doesMatShapeExist=1;
	    
	    double nphot=0.;
	    for(int is=0; is<_nsamplesshapes; is++) nphot+=laserShape[icol][iside]->GetBinContent(is+1);
	    if(nphot>0) laserShape[icol][iside]->Scale(1.0/nphot);
	  }
	}
	
      }
    }else{
      cout <<" ERROR! Matacq shape file not found ! ==> return "<< endl;
      return false;
    }
  }
  
  if (doesMatShapeExist==1) ok=true;
  
  return ok;
  
}

bool EcalLightAnalyzer::getShapeCor(int ieta, int iphi, int icol, double &shapeCor ) {


  // Get back Geometry to fill right tables
  //========================================
  
  shapeCor=1.;
  
  if( _type=="LED" ) return true; // LED SHAPES ARE FIXED
  
  int iside;
  if(_ecalPart=="EB"){
    iside=MEEBGeom::side(ieta,iphi);
  } else {
    int iX = (ieta-1)/5+1;
    int iY = (iphi-1)/5+1;
    iside=MEEEGeom::side(iX, iY, iZ);
  }
    
  // Initialize
  //============

  bool ok=false;
  
  double t0=_t0shapes;
  
  pair<double,double> taus;    
  taus=calibData->tauAPD( ieta, iphi );
  double tau1=taus.first;
  double tau2=taus.second;

  if(isMatacqOK){
    
    unsigned int nBins=int(pulseShape[icol][iside]->GetNbinsX());
    assert( _nsamplesshapes <= nBins);    

    double sprAPD_jj, laser_iiMinusjj;
    double sum_jj;
    
    double sprAPD_val[NSAMPSHAPES]; double sprAPD_max=0;
    for(unsigned int i=0; i<_nsamplesshapes; i++){
      sprAPD_val[i]=sprval(double(i+1), t0, tau1, tau2, 1.0, _fedid );
      if(sprAPD_val[i]>sprAPD_max) sprAPD_max=sprAPD_val[i];
    }
    
    for(unsigned int i=0; i<_nsamplesshapes; i++) {
      sprAPD_val[i]/=sprAPD_max;
    }
    
    double shapecor=0.;
    for(unsigned int ii=0;ii<_nsamplesshapes;ii++){
      sum_jj=0.0;
      for(unsigned int jj=0;jj<=ii;jj++){
	
	sprAPD_jj=sprAPD_val[jj];
	laser_iiMinusjj=pulseShape[icol][iside]->GetBinContent(ii-jj+1); 
	
	sum_jj+=sprAPD_jj*laser_iiMinusjj;
      }
      if(sum_jj>shapecor)shapecor=sum_jj;
    }
    shapeCor=shapecor;
    
    ok=true;
  }
  return ok;
}

bool EcalLightAnalyzer::getConvolution(int ieta, int iphi, int icol) {

  
  // Initialize
  //============
  
  int iside;
  if(_ecalPart=="EB") iside=MEEBGeom::side(ieta, iphi);
  else{
    int iX = (ieta-1)/5+1;
    int iY = (iphi-1)/5+1;
    iside=MEEEGeom::side(iX,iY,iZ);
  }

  bool ok=false;
  shapesAPD[icol][iside].clear();
  
  if(_debug==2 )cout<<" Getting convolution: "<< ieta<<" "<<iphi<<" "<<icol<<" "<<iside<< endl;


  double t0=_t0shapes;
  
  pair<double,double> taus;    
  double tau1;
  double tau2;
  
  taus=calibData->tauAPD( ieta,  iphi );
  tau1=taus.first;
  tau2=taus.second; 
  
  if(isMatacqOK){
    
    unsigned int nBins=int(pulseShape[icol][iside]->GetNbinsX());
    assert( _nsamplesshapes <= nBins);    
    
    double sprAPD_jj, laser_iiMinusjj;
    double sum_jj;
    
    if(_debug==2 ) cout <<"tau1="<< tau1<<" tau2="<<tau2<< endl;

    double sprAPD_val[NSAMPSHAPES]; double sprAPD_max=0;
    for(unsigned int i=0; i<_nsamplesshapes; i++){
      sprAPD_val[i]=sprval(double(i+1), t0, tau1, tau2, 1.0, _fedid );
      //      cout<< " sprval: " <<i<<"  " <<sprAPD_val[i]<<endl;
      if(sprAPD_val[i]>sprAPD_max) sprAPD_max=sprAPD_val[i];
    }
    //    cout<< " sprAPD_max " <<sprAPD_max<< endl;
    
    double maxShape=0.0; int iMaxShape=0;
    
    for(unsigned int ii=0;ii<_nsamplesshapes;ii++){
      sum_jj=0.0;
      for(unsigned int jj=0;jj<=ii;jj++){
	sprAPD_jj=sprAPD_val[jj]/sprAPD_max;
	laser_iiMinusjj=pulseShape[icol][iside]->GetBinContent(ii-jj+1);
	sum_jj+=sprAPD_jj*laser_iiMinusjj;
      }
      if(sum_jj>maxShape) {
	maxShape=sum_jj;
	iMaxShape=ii;
      }
      shapes[ii]=sum_jj;
    }
    
    if(_debug==2 ) cout<<" Getting convolution maxShape: "<< maxShape << " "<<iMaxShape<<endl;
    
    if(maxShape!=0.) {
      for(unsigned int ii=0;ii<_nsamplesshapes;ii++){
	shapesAPD[icol][iside].push_back(shapes[ii]/maxShape);
      }
      ok=true;
    }else{
      ok=false;
    }
  }
  return ok;
}

//========================================================================
double EcalLightAnalyzer::sprmax( double t0, double t1, double t2, double tau1,
				  double tau2, int fed ) {
//========================================================================
  
  TF1 *sprfunc=spr( t0, t1, t2, tau1, tau2, fed) ;
  
  double max=0.0;
  max=sprfunc->GetMaximum(t1,t2);

  delete sprfunc;
  return max;

}

//========================================================================
double EcalLightAnalyzer::sprval( double t, double t0, double tau1, 
				  double tau2, double max, double fed ) {
//========================================================================
  
  if (t<t0) return 0.0;
  
  if(fed<=609 ||fed>=646)
    {
      double a=tau1*tau2/(tau1-tau2);
      return (a/max)*(-exp(-(t-t0)/tau1)+exp(-(t-t0)/tau2))+(t-t0)*exp(-(t-t0)/tau1);
    }
  else
    {
      assert(max>0);
      return (t-t0)/tau1*exp(1-(t-t0)/tau1)/max;
    }
}

double EcalLightAnalyzer::sprvalled( double t, double t0, double tau1, 
				     double tau2, double max, double fed ) {
  
  if(t<t0) return 0.0;
  else if(t<tau1+t0) return 1.0;
  else if(t<250.) return exp(-((double)t-tau1-t0)/tau2);
  else return 0.;
}
//========================================================================
TF1* EcalLightAnalyzer::spr( double t0, double t1, double t2, double tau1,
				  double tau2, int fed ) {
//========================================================================

  TF1 *sprfunc;

  if(fed<=609 || fed>=646){
    sprfunc=new TF1("sprfunc","([0]*[1]/([0]-[1]))*(-exp(-(x-[2])/[0]) + exp(-(x-[2])/[1]))+ (x-[2])*exp(-(x-[2])/[0])", t1, t2);
    sprfunc->SetParameter(0,tau1);
    sprfunc->SetParameter(1,tau2);
    sprfunc->SetParameter(2,t0);
  }else{
    
    sprfunc=new TF1("sprfunc","((x-[1])/[0])*exp(1-(x-[1])/[0])", t1, t2);
    sprfunc->SetParameter(0,tau1);
    sprfunc->SetParameter(1,t0); 
  }
  return sprfunc;

}

void EcalLightAnalyzer::getLaserShapeForPN(int icol, int iside ) {
  
  if( isMatacqOK ){
    
    // For PNs, step is 25 ns :
    for(int is=0; is<50; is++) lasShapesForPN[icol][iside][is]=0.;
    for(int is=0; is<10; is++)
      {
        for(int i=0; i<25; i++) lasShapesForPN[icol][iside][is]+=(pulseShape[icol][iside]->GetBinContent(is*25+i));	
      }
  }
  gotLasShapeForPN[icol][iside]=true;
}

double EcalLightAnalyzer::getConvolutionPN(int iPN, int imem, int icol, int iside ) {
  
  
  if (_ecalPart == "EB") {
    if( _corrPNEB[iPN][icol][iside]>0.0 ) return _corrPNEB[iPN][icol][iside];
  }else{
    if( _corrPNEE[iPN][imem][icol][iside]>0.0 ) return _corrPNEE[iPN][imem][icol][iside];
  }

  if(!gotLasShapeForPN[icol][iside]){
    getLaserShapeForPN(icol,iside);
  }
  
  double sprPN_val[NSAMPPN];
  pair<double,double> tauspn=calibData->tauPN( iPN, imem );
  //double qmaxshape=calibData->qmaxPN( iPN, imem );
  double tau1=tauspn.first;
  double tau2=tauspn.second;
  double a=tau2/(tau2-tau1);
  //double a=tau2/(qmaxshape*(tau2-tau1));
  
  for(int i=0; i<NSAMPPN; i++)
    {
      double t=(double)i;
      double t0=1.;
      sprPN_val[i]=0.;
      if(t>t0)sprPN_val[i]=(a*(1.-a)*(exp(-(t-t0)/tau1)-exp(-(t-t0)/tau2))+
			  (t-t0)/tau1*(1.-a-(t-t0)/2./tau1)*exp(-(t-t0)/tau1));
      
    }
  double sprPN_max=0.;
  for(int i=0; i<NSAMPPN; i++) if(sprPN_val[i]>sprPN_max)sprPN_max=sprPN_val[i];
  for(int i=0; i<NSAMPPN; i++) sprPN_val[i]/=sprPN_max;
  
  int imax=0;
  double qmax=0.;
  for(int i=0; i<NSAMPPN; i++)
    {
      double pn_shape=0.;
      for(int j=0; j<=i; j++)
	{
	  pn_shape+=sprPN_val[j]*lasShapesForPN[icol][iside][i-j];
	}
      if(pn_shape>qmax)
	{
	  qmax=pn_shape;
	  imax=i;
	}
    }
  
  if (_ecalPart == "EB") {
    _corrPNEB[iPN][icol][iside]=qmax;
  }else{
    _corrPNEE[iPN][imem][icol][iside]=qmax;
  }
  return qmax;

}


//========================================================================
bool EcalLightAnalyzer::getAB() {
  //========================================================================

  bool ok=true;
  
  // check ab file exists
  //=======================
  stringstream nameabfile;  
  if(!_fitab) 
    nameabfile << alphainitpath_ <<"/AB"<<_fedid<<"_"<<_type<<".root";
  else
    nameabfile << resdir_ <<"/AB"<<_fedid<<"_"<<_type<<".root";

  alphafile=nameabfile.str();
  if ( _debug == 1 ) cout << "-- debug EcalLightAnalyzer -- getAB doesAB 1 "<<ok<< endl;
  if ( _debug == 1 ) cout << "-- debug EcalLightAnalyzer -- abfile:"<<alphafile<< endl;
  
  FILE *test;
  test = fopen(alphafile.c_str(),"r"); 
  if(!test) {
    ok=false;
    cout<< "AB TREE not found ==> return "<<endl;
    return false;
  }else{
    fclose(test);
  }
  
  if ( _debug == 1 ) cout << "-- debug EcalLightAnalyzer -- getAB doesAB 2 "<<ok<< " "<<alphafile<<endl;
  
  calibData->setABFile(alphafile);
  
  return ok;
}



  
DEFINE_FWK_MODULE(EcalLightAnalyzer);

