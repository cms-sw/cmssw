/* 
 *  \class EcalCalibAnalyzer
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

#include "CalibCalorimetry/EcalLaserAnalyzer/plugins/EcalCalibAnalyzer.h"

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
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TPN.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TPNPulse.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TPNCor.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMem.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TSFit.h>  
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/PulseFitWithFunction.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/PulseFitWithShape.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/ME.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/MEGeom.h>


//========================================================================
EcalCalibAnalyzer::EcalCalibAnalyzer(const edm::ParameterSet& iConfig)
//========================================================================
  :
iEvent(0),

// Framework parameters with default values
 
_type    (        iConfig.getUntrackedParameter< std::string  >( "type",          "LASER" ) ), 
_typefit    (     iConfig.getUntrackedParameter< std::string  >( "typefit",          "AB" ) ), 

// COMMON:
_nsamples(        iConfig.getUntrackedParameter< unsigned int >( "nSamples",           10 ) ),
_presample(       iConfig.getUntrackedParameter< unsigned int >( "nPresamples",         2 ) ),
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
_niter(           iConfig.getUntrackedParameter< unsigned int >( "nIter",               3 ) ),
_chi2cut(         iConfig.getUntrackedParameter< double       >( "chi2cut",          10.0 ) ), 
_ecalPart(        iConfig.getUntrackedParameter< std::string  >( "ecalPart",         "EB" ) ),
_fedid(           iConfig.getUntrackedParameter< int          >( "fedID",            -999 ) ),
_saveallevents(   iConfig.getUntrackedParameter< bool         >( "saveAllEvents",   false ) ),
_debug(           iConfig.getUntrackedParameter< int          >( "debug",              0  ) ),

// LASER/LED:
_noise(           iConfig.getUntrackedParameter< double       >( "noise",             2.0 ) ),
_docorpn(         iConfig.getUntrackedParameter< bool         >( "doCorPN",         false ) ),

// LASER/LED AB:
_fitab(           iConfig.getUntrackedParameter< bool         >( "fitAB",           false ) ),
_alpha(           iConfig.getUntrackedParameter< double       >( "alpha",       1.5076494 ) ),
_beta(            iConfig.getUntrackedParameter< double       >( "beta",        1.5136036 ) ),
_nevtmax(         iConfig.getUntrackedParameter< unsigned int >( "nEvtMax",           200 ) ),

// LASER/LED SHAPE:

_saveshapes(      iConfig.getUntrackedParameter< bool         >( "saveShapes",       true ) ),
_nsamplesshapes(                                                                NSAMPSHAPES ), 

// TESTPULSE: 

_samplemin(     iConfig.getUntrackedParameter< unsigned int >( "sampleMin",       3 ) ),
_samplemax(     iConfig.getUntrackedParameter< unsigned int >( "sampleMax",       9 ) ),
_chi2max(       iConfig.getUntrackedParameter< double       >( "chi2Max",      10.0 ) ),
_timeofmax(     iConfig.getUntrackedParameter< double       >( "timeOfMax",     4.5 ) ),

// COMMON:
nCrys(                                                                               NCRYSEB),
nPNPerMod(                                                                         NPNPERMOD),
nMod(                                                                                 NMODEE),
nSides(                                                                               NSIDES),
runType(-1), runNum(0), fedID(-1), dccID(-1), side(2), lightside(2), iZ(1),
phi(-1), eta(-1), event(0), color(-1),pn0(0), pn1(0), apdAmpl(0), apdAmplA(0), 
apdAmplB(0),apdTime(0),pnAmpl(0),
pnID(-1), moduleID(-1), channelIteratorEE(0),

// LASER/LED SHAPE:

isMatacqOK(false), ShapeCor(0),

// TEST PULSE:

nGainPN(                                                                       NGAINPN),
nGainAPD(                                                                     NGAINAPD)

//========================================================================

{

  // Initialization from cfg file

  resdir_                 = iConfig.getUntrackedParameter<std::string>("resDir");
  elecfile_               = iConfig.getUntrackedParameter<std::string>("elecFile", "ElecMeanShape.root");
  pncorfile_              = iConfig.getUntrackedParameter<std::string>("pnCorFile");

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

  if( _type != "LASER" && _type != "TESTPULSE" && _type!= "LED" ){
    cout << "Error: Wrong type "<<_type<<", should be LASER or LED or TESTPULSE"<< endl;
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
    iEta[j]=-1;
    iPhi[j]=-1;
    iModule[j]=10;
    iTowerID[j]=-1;
    iChannelID[j]=-1;
    idccID[j]=-1;
    iside[j]=-1;
    wasTimingOK[j]=true;
    wasGainOK[j]=true;
    wasSignalOK[j]=true;
    meanRawAmpl[j]=0.0;
    nEvtRawAmpl[j]=0;
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

  // PN linearity corrector

  pnCorrector = new TPNCor(pncorfile_.c_str());


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
EcalCalibAnalyzer::~EcalCalibAnalyzer(){
  //========================================================================


  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)

}



//========================================================================
void EcalCalibAnalyzer::beginJob() {
  //========================================================================

  
  // Create temporary files and trees to save adc samples
  //======================================================

  ADCfile=resdir_;
  ADCfile+="/APDSamples";
  ADCfile+=_type;
  ADCfile+=".root";

  APDfile=resdir_;
  APDfile+="/APDAmplitudes";  
  APDfile+=_type;
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
    ADCTrees[i]->SetBranchAddress( "pnGain",      &pnGain      );    
 
    nevtAB[i]=0 ;

  } 
 
  if ( _debug == 1 ) cout << "-- debug --  beginJob - _type = "<<_type<<" _typefit = "<<_typefit<< endl;
  if( _type == "LASER" || _type== "LED" ){
    if(_typefit == "AB") beginJobAB();
    if(_typefit == "SHAPE") beginJobShape();
  }
  

  // APD results file
  
  stringstream nameapdfile;
  nameapdfile << resdir_ <<"/APDPN_"<<_type<<".root";      
  resfile=nameapdfile.str();
  
  
  // Good type events counter
  
  typeEvents=0;
  
}


//========================================================================
void EcalCalibAnalyzer::beginJobAB() {
  //========================================================================


  if ( _debug == 1 ) cout << "-- debug -- beginJobAB"<< endl;
  
  // AlphaBeta files
  
  doesABTreeExist=true;
  doesABInitTreeExist=true;
  
  if ( _debug == 1 ) cout << "-- debug -- beginJobAB fitAB 1 "<<_fitab<< endl;

  // check ab file exists
  //=======================
  stringstream nameabfile;  
  nameabfile << resdir_ <<"/AB_"<<_type<<".root";
  alphafile=nameabfile.str();
  if ( _debug == 1 ) cout << "-- debug -- beginJobAB doesAB 1 "<<doesABTreeExist<< endl;
  
  FILE *test;
  test = fopen(alphafile.c_str(),"r"); 
  if(!test) {
    doesABTreeExist=false;
  }else{
    fclose(test);
  }
  if ( _debug == 1 ) cout << "-- debug -- beginJobAB doesAB 2 "<<doesABTreeExist<< endl;
  delete test;
  
  // if fitab = false check ab file exists and fit if not
  //=======================================================
  
  if(!_fitab && !doesABTreeExist){
    _fitab=true;
  }
  if ( _debug == 1 ) cout << "-- debug -- beginJobAB fitAB 2 "<<_fitab<< endl;

  // if fitab = false and ABfile exists, then fair enought
  //======================================================

  stringstream nameabinitfile;
  if(!_fitab){     
    nameabinitfile << resdir_ <<"/AB_"<<_type<<".root";
    alphainitfile=nameabinitfile.str();
  }else{
  // else look at initialisation file
  //==================================
    nameabinitfile << resdir_ <<"/AB_"<<_type<<"Init.root";
    alphainitfile=nameabinitfile.str();
  }
  if ( _debug == 1 ) cout << "-- debug -- beginJobAB doesAB3 "<< doesABTreeExist<<endl;
  if ( _debug == 1 ) cout << "-- debug -- beginJobAB doesABInit1 "<< doesABInitTreeExist<<endl;

  FILE *testinit;
  testinit = fopen(alphainitfile.c_str(),"r");   
  if(!testinit) {
    doesABInitTreeExist=false;
  }else{
    fclose(testinit);
  }
  delete testinit;  
  if ( _debug == 1 ) cout << "-- debug -- beginJobAB doesABInit2 "<< doesABInitTreeExist<<endl;
  
  TFile *fABInit=0; TTree *ABInit=0;
  if( doesABInitTreeExist){
    fABInit=new TFile(nameabinitfile.str().c_str());
    if(fABInit){
      ABInit = (TTree*) fABInit->Get("ABCol0");
    }
  }
  if ( _debug == 1 ) cout << "-- debug -- beginJobAB doesABInit3 "<< doesABInitTreeExist<<endl;
  
  
  // 2) Shape analyzer
  
  if(doesABInitTreeExist && fABInit && ABInit && ABInit->GetEntries()!=0){
    if ( _debug == 1 ) cout << "-- debug -- beginJobAB here1"<<endl;
    for(unsigned int ic=0;ic<nColor;ic++){
      shapana[ic]= new TShapeAnalysis(ABInit, _alpha, _beta, 5.5, 1.0, ic);
    }
  }else{
    if ( _debug == 1 ) cout << "-- debug -- beginJobAB here2"<<endl;
    for(unsigned int ic=0;ic<nColor;ic++){
      shapana[ic]= new TShapeAnalysis(_alpha, _beta, 5.5, 1.0, ic);
    }
    _fitab=true;  
  }  
  if ( _debug == 1 ) cout << "-- debug -- beginJobAB fitAB 3 "<<_fitab<< endl;

  for(unsigned int ic=0;ic<nColor;ic++){
    shapana[ic] -> set_const(_nsamples,_firstsample,_lastsample,
			     _presample, _nevtmax, _noise, _chi2cut);
  }
  
  if(doesABInitTreeExist && fABInit ) fABInit->Close();
}

//========================================================================
void EcalCalibAnalyzer::beginJobShape() {
  //========================================================================
 

  if ( _debug == 1 ) cout << "-- debug -- beginJobShape"<< endl;

  // Define output results filenames 
  //==================================
  stringstream namefile1;
  namefile1 << resdir_ <<"/SHAPE_"<<_type<<".root";      
  shapefile=namefile1.str();
  
  stringstream namefile3;
  namefile3 << resdir_ <<"/MATACQ.root";      
  matfile=namefile3.str();

}

//========================================================================
void EcalCalibAnalyzer:: analyze( const edm::Event & e, const  edm::EventSetup& c){
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
    return;
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
    else if(_type == "TESTPULSE"
	    && runType!=EcalDCCHeaderBlock::TESTPULSE_MGPA 
	    && runType!=EcalDCCHeaderBlock::TESTPULSE_GAP 
	    && runType!=EcalDCCHeaderBlock::TESTPULSE_SCAN_MEM ) return;

    // Retrieve laser color and event number
    
    if( _type != "TESTPULSE" ){  
      
      EcalDCCHeaderBlock::EcalDCCEventSettings settings = headerItr->getEventSettings();
      color = settings.wavelength;
      if( color<0 ) return;
      
      vector<int>::iterator iter = find( colors.begin(), colors.end(), color );
      if( iter==colors.end() ){
	colors.push_back( color );
	//cout <<" new color found "<< color<<" "<< colors.size()<< endl;
      }
      
    }else{
      color=0;
      colors.push_back( color );
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
  unsigned int samplemax=0;
  pnGain=0;
  
  map <int, vector<double> > allPNAmpl;
  map <int, vector<double> > allPNGain;

  
  // Loop on PNs digis
  
  for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNDigi->begin();
	pnItr != PNDigi->end(); ++pnItr ) { 
    
    EcalPnDiodeDetId pnDetId = EcalPnDiodeDetId((*pnItr).id());
    
    if ( _debug == 2 ) cout <<"-- debug -- analyze: in PNDigi - pnID=" <<
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
    chi2pn = pnfit -> doFit(samplemax,&pnNoPed[0]); 
    if(chi2pn == 101 || chi2pn == 102 || chi2pn == 103) pnAmpl=0.;
    else pnAmpl= pnfit -> getAmpl();
    
    // Apply linearity correction

    double corr=1.0;
    if( _docorpn ) corr=pnCorrector->getPNCorrectionFactor(pnAmpl, pnGain);
    pnAmpl*=corr;

    // Fill PN ampl vector

    allPNAmpl[pnDetId.iDCCId()].push_back(pnAmpl);
      
    if ( _debug == 2 ) cout <<"-- debug -- analyze: in PNDigi - PNampl=" << 
		     pnAmpl<<", PNgain="<< pnGain<<endl;  
  }
  
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
           
      int apdRefTT=MEEBGeom::apdRefTower(module);      
           
      std::pair<int,int> pnpair=MEEBGeom::pn(module,_fedid);
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
  
      iEta[channel]=eta;
      iPhi[channel]=phi;
      iModule[channel]= module ;
      iTowerID[channel]=towerID;
      iChannelID[channel]=channelID;
      idccID[channel]=dccID;
      iside[channel]=side;     
      
      if ( _debug == 2 ) cout << "-- debug -- analyze: in EBDigi - towerID:"<< towerID<<
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

      if(allPNAmpl[mem0].size()>MyPn0) pn0=allPNAmpl[mem0][MyPn0];
      else pn0=0;
      if(allPNAmpl[mem1].size()>MyPn1) pn1=allPNAmpl[mem1][MyPn1];
      else pn1=0;


      // Fill if Pulse is fine
      //=======================

      meanRawAmpl[channel]+=(APDPulse->getMax()-APDPulse->getPedestal());
      nEvtRawAmpl[channel]++;
      
      if( APDPulse->isPulseOK() ){
	
	ADCTrees[channel]->Fill(); 
	
	Delta01->addEntry(APDPulse->getDelta(0,1));
	Delta12->addEntry(APDPulse->getDelta(1,2));
	
	if( nevtAB[channel] < _nevtmax && _fitab && _typefit == "AB"){
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
       
      int etaG = id_crystal.ix() ; 
      int phiG = id_crystal.iy() ;

      int iY = (phiG-1)/5+1;
      int iX = (etaG-1)/5+1;

      side=MEEEGeom::side(iX, iY, iZ);

      int tower=elecid_crystal.towerId();
      int ch=elecid_crystal.channelId()-1; 

      int module=MEEEGeom::lmmod( iX, iY );
      if( module>=18 && side==1 ) module+=2;
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
      
      assert( module>=*min_element(modules.begin(),modules.end()) 
	      && module<=*max_element(modules.begin(),modules.end()) );
      
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
      } 
      
      iEta[channel]=eta;
      iPhi[channel]=phi;
      iModule[channel]= module ;
      iTowerID[channel]=towerID;
      iChannelID[channel]=channelID;
      idccID[channel]=dccID;
      iside[channel]=side;
            
      assert ( channel < nCrys );
      
      if ( _debug == 2 ) cout << "-- debug -- analyze: in EEDigi - towerID:"<< towerID<<
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

      
      // Associate PN ampl
      //===================

      int mem0=Mem->Mem(lmr,0);
      int mem1=Mem->Mem(lmr,1);
      
      if(allPNAmpl[mem0].size()>MyPn0) pn0=allPNAmpl[mem0][MyPn0];
      else pn0=0;
      if(allPNAmpl[mem1].size()>MyPn1) pn1=allPNAmpl[mem1][MyPn1];
      else pn1=0;

      // Fill if Pulse is fine
      //=======================
      
      meanRawAmpl[channel]+=(APDPulse->getMax()-APDPulse->getPedestal());
      nEvtRawAmpl[channel]++;

      if( APDPulse->isPulseOK()){
	ADCTrees[channel]->Fill(); 
	
	Delta01->addEntry(APDPulse->getDelta(0,1));
	Delta12->addEntry(APDPulse->getDelta(1,2));
	
	if( nevtAB[channel] < _nevtmax && _fitab && _typefit == "AB" ){

	  shapana[iCol] -> putAllVals(channel, adc, eta, phi, dccID, side, towerID, channelID); 
	  nevtAB[channel]++ ;
	}
      }
    }
  }
}

void EcalCalibAnalyzer::endJob() {
  
  
  // Don't do anything if there is no events
  //=========================================

  if( typeEvents == 0 ){
    
    ADCFile->Close(); 
    stringstream del;
    del << "rm " <<ADCfile;
    system(del.str().c_str());

    cout << "\n\t***  No "<< _type<< " Events  ***"<< endl;
 
    return;
  }

  if ( _debug == 1 ) cout << "-- debug -- endJob - still Here"<< endl;
  
  // Adjust channel numbers for EE 
  //===============================

  if( _ecalPart == "EE" ) {
    nCrys=channelMapEE.size();
  }
  if ( _debug == 1 ) cout << "-- debug -- endJob - nCrys = "<< nCrys << endl;
  
  // Set presamples number 
  //======================
  
  double delta01=Delta01->getMean();
  double delta12=Delta12->getMean();
  if(delta12>_presamplecut) {
    _presample=2;
    if(delta01>_presamplecut) _presample=1;
  }
  
  APDPulse->setPresamples(_presample);

  if ( _debug == 1 ) cout << "-- debug -- endJob - _presample = "<< _presample << endl;


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


  if ( _debug == 1 ) cout << "-- debug -- endJob - isGainOK = "<< 
		       isGainOK << " isTimingOK = "<<isTimingOK<< endl;

  if( _type == "LASER" || _type == "LED")
    {
      if( _typefit == "AB" ) endJobAB();
      if( _typefit == "SHAPE" ) endJobShape();
      endJobLight();
    }
  if( _type == "TESTPULSE") endJobTestPulse();
  
  
}

//========================================================================
void EcalCalibAnalyzer::endJobAB() {
//========================================================================


  if ( _debug == 1 ) cout << "-- debug -- endJobAB"<< endl;

  // Adjust channel numbers for EE 
  //===============================

  if( _ecalPart == "EE" ) {
    for(unsigned int ic=0;ic<colors.size();ic++){
      shapana[ic]->set_nch(nCrys);
    }
  }

    
  if(_fitab){
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
      TFile *fAB=0; TTree *ABInit=0;
      if(doesABInitTreeExist){
	fAB=new TFile(alphainitfile.c_str());
      }
      if(doesABInitTreeExist && fAB ){
	ABInit = (TTree*) fAB->Get("ABCol0");
      }
      shapana[ic]->computeShape(alphafile, ABInit);  
    }
    cout <<    "\t+=+       .................................... done     +=+" << endl;
    cout <<    "\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;
  }
}

void EcalCalibAnalyzer::endJobLight() {

  
  if ( _debug == 1 ) cout << "-- debug -- endJobLight"<< endl;

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

  if( _typefit == "AB" ) psfit = new PulseFitWithFunction();
  else if( _typefit == "SHAPE" ) psfit = new PulseFitWithShape();
  else psfit=0;

  double chi2;

  for (unsigned int iCry=0;iCry<nCrys;iCry++){

    for (unsigned int icol=0;icol<nCol;icol++){

      // Declare APD stuff
      //===================
      
      APDFirstAnal[iCry][icol] = new TAPD();
      isThereDataADC[iCry][icol]=1;      
      stringstream cut;
      cut <<"color=="<<colors[icol];
      if(ADCTrees[iCry]->GetEntries(cut.str().c_str())<10) isThereDataADC[iCry][icol]=0;  
      
    }

    unsigned int iMod=iModule[iCry]-1;
    
    
    // Initialize Pulse Fit 
    //======================

    if( _typefit == "SHAPE" ) {
      
      if ( _debug == 1 ) cout << " -- debug -- endJobLight - crys = "<< iCry<<" shapesVec size: "<< shapesVec.size()<< " 0:"<< shapesVec[0] << " 1/2:"<< shapesVec[shapesVec.size()/2]<< " 2/2:" << shapesVec[shapesVec.size()-1]<< endl;
      
      if ( _debug == 1 ) cout << " -- debug -- endJobLight - isSPRFine "<< isSPRFine<< endl;
      
      if(isSPRFine) ((PulseFitWithShape*) psfit) -> init(_nsamples,_firstsample,_lastsample,_niter, _nsamplesshapes,_presample, shapesVec, _noise );
      else{ 
	cout << "Wrong SPR function: analysis aborted"<< endl; 
	return;
      }
    } 
    
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
      
    
    // Initialize Pulse Fit 
    //======================

    double alpha, beta;
      if( _typefit == "AB" ){
	
	std::vector<double> abvals=shapana[iCol]->getVals(iCry);	
	alpha=abvals[0];
	beta=abvals[1];  
	if ( _debug == 1 ) cout << " -- debug -- endJobLight - crys = "<< iCry<<" alpha = "<< alpha<<" beta = "<< beta<< endl;
	
	((PulseFitWithFunction*) psfit) -> init(_nsamples,_firstsample,_lastsample,_niter,  alpha,  beta);
	
      }

      //assert( iPhi[iCry]-phi == 0 && iEta[iCry]-eta == 0 && 
      // towerID -iTowerID[iCry] == 0 && channelID-iChannelID[iCry] == 0 );
      
      // Amplitude calculation
      
      APDPulse->setPulse(adc);
      adcNoPed=APDPulse->getAdcWithoutPedestal();

      apdAmpl=0;
      apdAmplA=0;
      apdAmplB=0;
      apdTime=0;
      
      if( APDPulse->isPulseOK()){
	
	chi2 = psfit -> doFit(&adcNoPed[0]);
	
	apdAmpl = psfit -> getAmpl();
	apdTime = psfit -> getTime();
	flag=1;

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
      
      if ( _debug == 2 ) cout <<"-- debug -- endJobLight - apdAmpl="<<apdAmpl<< 
			   ", apdTime="<< apdTime<<" "<< chi2<<" " << flag<< endl; 

      if ( _debug == 2 ) cout <<"-- debug -- endJobLight - pn0="<<pn0<<", pn1="<<pn1<< endl;
      
      // Fill PN stuff
      //===============
      
      if( firstChanMod[iMod]==iCry && isThereDataADC[iCry][iCol]==1 ){ 
	for (unsigned int ichan=0;ichan<nPNPerMod;ichan++){
	  PNFirstAnal[iMod][ichan][iCol]->addEntry(pn0,pn1);
	}
      }
      
      // Fill APD stuff 
      //================

      if( APDPulse->isPulseOK() ){
	APDFirstAnal[iCry][iCol]->addEntry(apdAmpl, pn0, pn1, apdTime);      
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
    resTrees[iColor]->Branch( "APDoAPD",     &APDoAPD,     "APDoAPD[6]/D"    );
    resTrees[iColor]->Branch( "APDoAPDA",    &APDoAPDA,    "APDoAPDA[6]/D"    );
    resTrees[iColor]->Branch( "APDoAPDB",    &APDoAPDB,    "APDoAPDB[6]/D"    );
    resTrees[iColor]->Branch( "flag",        &flag,        "flag/I"           ); 

    resPNTrees[iColor]->Branch( "side",      &side,      "side/I"         );
    resPNTrees[iColor]->Branch( "moduleID",  &moduleID,  "moduleID/I"     );
    resPNTrees[iColor]->Branch( "pnID",      &pnID,      "pnID/I"         );
    resPNTrees[iColor]->Branch( "PN",        &PN,        "PN[6]/D"        ); 
    resPNTrees[iColor]->Branch( "PNoPN",     &PNoPN,     "PNoPN[6]/D"     );
    resPNTrees[iColor]->Branch( "PNoPNA",    &PNoPNA,    "PNoPNA[6]/D"    );
    resPNTrees[iColor]->Branch( "PNoPNB",    &PNoPNB,    "PNoPNB[6]/D"    );

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
    resTrees[iColor]->SetBranchAddress( "APDoAPD",     APDoAPD     );
    resTrees[iColor]->SetBranchAddress( "APDoAPDA",    APDoAPDA    );
    resTrees[iColor]->SetBranchAddress( "APDoAPDB",    APDoAPDB    );
    resTrees[iColor]->SetBranchAddress( "flag",        &flag       ); 
   
    resPNTrees[iColor]->SetBranchAddress( "side",      &side     );
    resPNTrees[iColor]->SetBranchAddress( "moduleID",  &moduleID );
    resPNTrees[iColor]->SetBranchAddress( "pnID",      &pnID     );
    resPNTrees[iColor]->SetBranchAddress( "PN",        PN        );
    resPNTrees[iColor]->SetBranchAddress( "PNoPN",     PNoPN     );
    resPNTrees[iColor]->SetBranchAddress( "PNoPNA",    PNoPNA    );
    resPNTrees[iColor]->SetBranchAddress( "PNoPNB",    PNoPNB    );
  
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
    
    unsigned int iMod=iModule[iCry]-1;
    
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
      
    }

    // Final loop on events
    //=======================

    Long64_t nbytes = 0, nb = 0;
    for (Long64_t jentry=0; jentry< APDTrees[iCry]->GetEntriesFast();jentry++) { 
      nb = APDTrees[iCry]->GetEntry(jentry);   nbytes += nb; 
      
      // Get back color
      //================
      
      unsigned int iCol=0;
      for(unsigned int i=0;i<nCol;i++){
	if(color==colors[i]) {
	  iCol=i;
	  i=colors.size();
	}
      }
      
      // Fill PN stuff
      //===============
      
      if( firstChanMod[iMod]==iCry && isThereDataADC[iCry][iCol]==1 ){ 
	for (unsigned int ichan=0;ichan<nPNPerMod;ichan++){
	  PNAnal[iMod][ichan][iCol]->addEntry(pn0,pn1);
	}
      }
      
      // Get ref amplitudes
      //===================

      if ( _debug == 2 ) cout <<"-- debug -- endJobLight - Last Loop event:"<<event<<" apdAmpl:"<< apdAmpl<< endl;
      apdAmplA = 0.0;
      apdAmplB = 0.0;
      
      for (unsigned int iRef=0;iRef<nRefChan;iRef++){
	refAPDTrees[iRef][iMod+1]->GetEntryWithIndex(event); 
      }
      
      if ( _debug == 2 ) cout <<"-- debug -- endJobLight - Last Loop apdAmplA:"<<apdAmplA<< " apdAmplB:"<< apdAmplB<<", event:"<< event<<", eventref:"<< eventref<< endl;
      
      
      // Fill APD stuff
      //===============
      
      APDAnal[iCry][iCol]->addEntry(apdAmpl, pn0, pn1, apdTime, apdAmplA, apdAmplB); 
   
    }
    
    moduleID=iMod+1;
    
    if( moduleID>=20 ) moduleID-=2; // Trick to fix endcap specificity   
    
    // Get final results for APD
    //===========================
    
    for(unsigned int iColor=0;iColor<nCol;iColor++){
      
      std::vector<double> apdvec = APDAnal[iCry][iColor]->getAPD();
      std::vector<double> apdpnvec = APDAnal[iCry][iColor]->getAPDoPN();
      std::vector<double> apdpn0vec = APDAnal[iCry][iColor]->getAPDoPN0();
      std::vector<double> apdpn1vec = APDAnal[iCry][iColor]->getAPDoPN1();
      std::vector<double> timevec = APDAnal[iCry][iColor]->getTime();
      std::vector<double> apdapdvec = APDAnal[iCry][iColor]->getAPDoAPD();
      std::vector<double> apdapd0vec = APDAnal[iCry][iColor]->getAPDoAPD0();
      std::vector<double> apdapd1vec = APDAnal[iCry][iColor]->getAPDoAPD1();
      
      
      for(unsigned int i=0;i<apdvec.size();i++){
	
	APD[i]=apdvec.at(i);
	APDoPN[i]=apdpnvec.at(i);
	APDoPNA[i]=apdpn0vec.at(i);
	APDoPNB[i]=apdpn1vec.at(i);
	APDoAPD[i]=apdapdvec.at(i);
	APDoAPDA[i]=apdapd0vec.at(i);
	APDoAPDB[i]=apdapd1vec.at(i);
	Time[i]=timevec.at(i);
      }
      
      
      // Fill APD results trees
      //========================

      phi=iPhi[iCry];
      eta=iEta[iCry];
      dccID=idccID[iCry];
      side=iside[iCry];
      towerID=iTowerID[iCry];
      channelID=iChannelID[iCry];

      
      if( !wasGainOK[iCry] || !wasTimingOK[iCry] || isThereDataADC[iCry][iColor]==0 ){
	flag=0;
      }else flag=1;
      
      resTrees[iColor]->Fill();   
    }
  }

  // Get final results for PN 
  //==========================
  
  for (unsigned int iM=0;iM<nMod;iM++){
    unsigned int iMod=modules[iM]-1;

    side=iside[firstChanMod[iMod]];
    
    for (unsigned int ch=0;ch<nPNPerMod;ch++){

      pnID=ch;
      moduleID=iMod+1;
      
      if( moduleID>=20 ) moduleID-=2;  // Trick to fix endcap specificity

      for(unsigned int iColor=0;iColor<nCol;iColor++){

	std::vector<double> pnvec = PNAnal[iMod][ch][iColor]->getPN();
	std::vector<double> pnopnvec = PNAnal[iMod][ch][iColor]->getPNoPN();
	std::vector<double> pnopn0vec = PNAnal[iMod][ch][iColor]->getPNoPN0();
	std::vector<double> pnopn1vec = PNAnal[iMod][ch][iColor]->getPNoPN1();
	
	for(unsigned int i=0;i<pnvec.size();i++){
	  
	  PN[i]=pnvec.at(i);
	  PNoPN[i]=pnopnvec.at(i);
	  PNoPNA[i]=pnopn0vec.at(i);
	  PNoPNB[i]=pnopn1vec.at(i);
	}
	
	// Fill PN results trees
	//========================

	resPNTrees[iColor]->Fill();
      }
    }
  }
  
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
  
  // Save results 
  //===============

  for (unsigned int i=0;i<nCol;i++){
    resTrees[i]->Write();
    resPNTrees[i]->Write();
  }  
  
  resFile->Close(); 
  
  cout <<    "\t+=+    .................................................. done  +=+" << endl;
  cout <<    "\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;
}

//========================================================================
void EcalCalibAnalyzer::endJobShape() {
//========================================================================

  if ( _debug == 1 ) cout << "-- debug test -- endJobShape"<< endl;
  
  // Get Pulse Shapes
  //==================
  
  isMatacqOK=getShapes();   
  if(!isMatacqOK){
    cout <<" ERROR! No matacq shape available: analysis aborted !"<< endl;
    return;
  }

  if(!isMatacqOK){
    
    cout << "\n\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;
    cout <<   "\t+=+     WARNING! NO MATACQ        +=+" << endl;
    cout <<   "\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;
    return;

  }

}
//========================================================================
void EcalCalibAnalyzer::endJobTestPulse() {
//========================================================================

  cout <<  "\n\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;
  cout <<    "\t+=+     Analyzing TESTPULSE data: getting APD, PN   +=+" << endl;
  
  
  
  // Create output trees
  //=====================
  
  int gain=0;

  resFile = new TFile(resfile.c_str(),"RECREATE");
  
  resTree= new TTree("TPAPD","TPAPD");
  resPNTree= new TTree("TPPN","TPPN");

  resTree->Branch( "iphi",        &phi,         "phi/I"           );
  resTree->Branch( "ieta",        &eta,         "eta/I"           );
  resTree->Branch( "dccID",       &dccID,       "dccID/I"          );
  resTree->Branch( "side",        &side,        "side/I"           );
  resTree->Branch( "towerID",     &towerID,     "towerID/I"        );
  resTree->Branch( "channelID",   &channelID,   "channelID/I"      );
  resTree->Branch( "moduleID",    &moduleID,    "moduleID/I"       );
  resTree->Branch( "flag",        &flag,        "flag/I"           );
  resTree->Branch( "gain",        &gain,        "gain/I"           );
  resTree->Branch( "APD",         &APD,         "APD[6]/D"         );
  
  resPNTree->Branch( "pnID",      &pnID,      "pnID/I"         );
  resPNTree->Branch( "moduleID",  &moduleID,  "moduleID/I"     );
  resPNTree->Branch( "gain",      &gain,      "gain/I"         ); 
  resPNTree->Branch( "PN",        &PN,        "PN[6]/D"        ); 
  
  resTree->SetBranchAddress( "iphi",        &phi       );
  resTree->SetBranchAddress( "ieta",        &eta       );
  resTree->SetBranchAddress( "dccID",       &dccID      );
  resTree->SetBranchAddress( "side",        &side       );
  resTree->SetBranchAddress( "towerID",     &towerID    );
  resTree->SetBranchAddress( "channelID",   &channelID  );
  resTree->SetBranchAddress( "moduleID",    &moduleID   ); 
  resTree->SetBranchAddress( "flag",        &flag       );  
  resTree->SetBranchAddress( "gain",        &gain       );  
  resTree->SetBranchAddress( "APD",         APD         );  
  
  resPNTree->SetBranchAddress( "pnID",      &pnID     );
  resPNTree->SetBranchAddress( "moduleID",  &moduleID );
  resPNTree->SetBranchAddress( "gain",      &gain     ); 
  resPNTree->SetBranchAddress( "PN",        PN        ); 
  

  for (unsigned int iMod=0;iMod<nMod;iMod++){
    for (unsigned int ich=0;ich<2;ich++){
      for (unsigned int ig=0;ig<nGainPN;ig++){
	TPPNAnal[iMod][ich][ig]=new TPN();
      }
    }
  }

  TSFit * pstpfit = new TSFit(_nsamples,650);
  pstpfit -> set_params(_nsamples, _niter, _presample, _samplemin, _samplemax, _timeofmax ,  _chi2max, _firstsample,  _lastsample);

  for (unsigned int iCry=0;iCry<nCrys;iCry++){

    for(unsigned int iG=0;iG<nGainAPD;iG++){
      TPAPDAnal[iCry][iG]=new TAPD();
    }
    
    unsigned int iMod=iModule[iCry]-1;
  
    moduleID=iMod+1;    
    if( moduleID>=20 ) moduleID-=2;  // Trick to fix endcap specificity
    
    Long64_t nbytes = 0, nb = 0;
    for (Long64_t jentry=0; jentry<  ADCTrees[iCry]->GetEntriesFast();jentry++) { 
      nb = ADCTrees[iCry]->GetEntry(jentry);   nbytes += nb; 
      
      pstpfit -> init_errmat(10.);

      // PN Means and RMS 
      //==================

      if( firstChanMod[iMod] == iCry ){ 
	for (unsigned int ichan=0;ichan<nPNPerMod;ichan++){
	  TPPNAnal[iMod][ichan][pnGain]->addEntry(pn0,pn1);
	}
      }
      

      // APD means and RMS
      //==================
      
      APDPulse->setPulse(adc);
      adcNoPed=APDPulse->getAdcWithoutPedestal();
      
      if( APDPulse->isPulseOK()){
	
	// Retrieve APD amplitude
	//=======================
	
	double res[20];

	double chi2 =  pstpfit -> fit_third_degree_polynomial(&adcNoPed[0],res);
	
	if( chi2 < 0. || chi2 == 102 ) {
	  
	  apdAmpl=0;
	  apdTime=0;
	  
	}else{
	  
	  apdAmpl = res[0];
	  apdTime = res[1];
	  
	}
	
	TPAPDAnal[iCry][adcGain]->addEntry(apdAmpl,pn0, pn1, apdTime);
      
      }
    }
    
    if (ADCTrees[iCry]->GetEntries()<10){
      flag=-1;
      for (int j=0;j<6;j++){
	APD[j]=0.0;
      }
    }
    else flag=1;
    
    phi=iPhi[iCry];
    eta=iEta[iCry];
    dccID=idccID[iCry];
    side=iside[iCry];
    towerID=iTowerID[iCry];
    channelID=iChannelID[iCry];

    for (unsigned int ig=0;ig<nGainAPD;ig++){
      
      std::vector<double> apdvec = TPAPDAnal[iCry][ig]->getAPD();
      std::vector<double> timevec = TPAPDAnal[iCry][ig]->getTime();
      
      for(unsigned int i=0;i<apdvec.size();i++){
	
	APD[i]=apdvec.at(i);
	Time[i]=timevec.at(i);
      }
      gain=ig;
      
      // Fill APD tree
      //===============
      
      if( APD[0]!=0 && APD[3]>10 ){
	resTree->Fill(); 
      }
    }     
  }
  
  // Get final results for PN and PN/PN
  //====================================
  
  for (unsigned int ig=0;ig<nGainPN;ig++){
    for (unsigned int iMod=0;iMod<nMod;iMod++){
      for (int ch=0;ch<2;ch++){
	
	pnID=ch;
	moduleID=iMod;
	if( moduleID>=20 ) moduleID-=2;  // Trick to fix endcap specificity
	std::vector<double> pnvec = TPPNAnal[iMod][ch][ig]->getPN();
	for(unsigned int i=0;i<pnvec.size();i++){
	  PN[i]=pnvec.at(i);
	}
	gain=ig;
	
	// Fill PN tree
	//==============
      
	if( PN[0]!=0 && PN[3]>10 ){
	  resPNTree->Fill();
	}
      }
    }
  }

  ADCFile->Close(); 
  
  // Remove temporary file
  //=======================
  stringstream del;
  del << "rm " <<ADCfile;
  system(del.str().c_str());
  


  if ( _debug == 1 ) cout << "-- debug test -- endJobTestPulse before to save"<<endl;

  // Save final results 
  //=======================

  resTree->Write();
  resPNTree->Write();
  resFile->Close(); 
  
  cout <<    "\t+=+    ...................................... done  +=+" << endl;
  cout <<    "\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;


}


//========================================================================
bool EcalCalibAnalyzer::getShapes() {
//========================================================================
  

  // Get Pulse From Matacq Analysis:
  //================================

  bool IsMatacqOK=false;

  int doesMatFileExist=0;
  int doesMatShapeExist=0;
  FILE *test2;   
  TProfile *laserShape=0;
  test2 = fopen(matfile.c_str(),"r");
  if (test2){
    doesMatFileExist=1; 
    fclose(test2);
  }
  TFile *matShapeFile;
  if (doesMatFileExist==1){
    matShapeFile = new TFile(matfile.c_str());
    laserShape= (TProfile*) matShapeFile->Get("shapeLaser");
    if(laserShape){
      doesMatShapeExist=1;
      double y=laserShape->Integral("w");
      if(y!=0)laserShape->Scale(1.0/y);
    }
  }else{

    cout <<" ERROR! Matacq shape file not found !"<< endl;
     
  }
  if (doesMatShapeExist) IsMatacqOK=true;

  // Get SPR from the average elec shape in SM6:
  //============================================

  int doesElecFileExist=0;
  FILE *test; 
  test = fopen(elecfile_.c_str(),"r");
  if (test){
    doesElecFileExist=1; 
    fclose(test);
  }
  TFile *ElecShapesFile;
  TH1D* elecShape=0 ;

  if (doesElecFileExist==1){
    ElecShapesFile = new TFile(elecfile_.c_str());
    stringstream name;
    name << "MeanElecShape";
    elecShape=(TH1D*) ElecShapesFile->Get(name.str().c_str());
    if(elecShape && doesMatShapeExist==1){
      double x=elecShape->GetMaximum();
      if (x!=0) elecShape->Scale(1.0/x);
      isSPRFine=true;
    }else{
      isSPRFine=false;
    }
    
  }else{
    
    cout <<" ERROR! Elec shape file not found !"<< endl;
    
  }
  

  if(IsMatacqOK){
    
    shapeFile = new TFile(shapefile.c_str(),"RECREATE");
    
    unsigned int nBins=int(laserShape->GetEntries());
    assert( _nsamplesshapes == nBins);
    double elec_jj, laser_iiMinusjj;
    double sum_jj;
    
    if(isSPRFine==true){
      
      unsigned int nBins2=int(elecShape->GetNbinsX());
      
      if(nBins2<nBins){  
	cout<< "getShapes: wrong configuration of the shapes' number of bins"<< std::endl;
	isSPRFine=false;
      }
      assert(_nsamplesshapes == nBins2 );
      
      stringstream name;
      name << "PulseShape";
      
      pulseShape=new TProfile(name.str().c_str(),name.str().c_str(),nBins,-0.5,double(nBins)-0.5);
      
      // shift shapes to have max close to the real APD max
      
      for(int ii=0;ii<50;ii++){
	shapes[ii]=0.0;
	pulseShape->Fill(ii,0.0);
      }
      
      for(unsigned int ii=0;ii<nBins-50;ii++){
	sum_jj=0.0;
	for(unsigned int jj=0;jj<ii;jj++){
	  elec_jj=elecShape->GetBinContent(jj+1);
	  laser_iiMinusjj=laserShape->GetBinContent(ii-jj+1);
	  sum_jj+=elec_jj*laser_iiMinusjj;
	}
	pulseShape->Fill(ii+50,sum_jj);
	shapes[ii+50]=sum_jj;
      }
      
      double scale= pulseShape->GetMaximum();
      shapeCorrection=scale;
      
      if(scale!=0) {
	pulseShape->Scale(1.0/scale);
	for(unsigned int ii=0;ii<nBins;ii++){
	  shapesVec.push_back(shapes[ii]/scale);
	}
      }
      
      if(_saveshapes) pulseShape->Write();
    }
  }
  shapeFile->Close();

  if(!_saveshapes) {

    stringstream del;
    del << "rm " <<shapefile;
    system(del.str().c_str()); 
    
  }
  
  return IsMatacqOK;
}
  
DEFINE_FWK_MODULE(EcalCalibAnalyzer);

