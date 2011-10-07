/* 
 *  \class EcalTestPulseAnalyzer
 *
 *  $Date: 2010/04/09 14:45:37 $
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

#include "CalibCalorimetry/EcalLaserAnalyzer/plugins/EcalTestPulseAnalyzer.h"

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
EcalTestPulseAnalyzer::EcalTestPulseAnalyzer(const edm::ParameterSet& iConfig)
//========================================================================
  :
iEvent(0),

// Framework parameters with default values
 
_type("TESTPULSE") , 

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
_ratiomincutlow(  iConfig.getUntrackedParameter< double       >( "ratioMinCutLow",    0.2 ) ),
_ratiomincuthigh( iConfig.getUntrackedParameter< double       >( "ratioMinCutHigh",   1.0 ) ),
_ratiomaxcutlow(  iConfig.getUntrackedParameter< double       >( "ratioMaxCutLow",    0.5 ) ),
_pulsemaxcutlow(  iConfig.getUntrackedParameter< double       >( "pulseMaxCutLow",   50.0 ) ),
_pulsemaxcuthigh( iConfig.getUntrackedParameter< double       >( "pulseMaxCutHigh", 20000.0 ) ),
_qualpercent(     iConfig.getUntrackedParameter< double       >( "qualPercent",       0.2 ) ),
_presamplecut(    iConfig.getUntrackedParameter< double       >( "presampleCut",      5.0 ) ),
_niter(           iConfig.getUntrackedParameter< unsigned int >( "nIter",               3 ) ),
_ecalPart(        iConfig.getUntrackedParameter< std::string  >( "ecalPart",         "EB" ) ),
_fedid(           iConfig.getUntrackedParameter< int          >( "fedID",            -999 ) ),
_debug(           iConfig.getUntrackedParameter< int          >( "debug",              0  ) ),


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
runType(-1), runNum(0), fedID(-1), dccID(-1), side(2),  iZ(1),
phi(-1), eta(-1), event(0), pn0(0), pn1(0), apdAmpl(0), apdAmplA(0), 
apdAmplB(0),apdTime(0),pnAmpl(0),
pnID(-1), moduleID(-1), channelIteratorEE(0),

// TEST PULSE:

nGainPN(                                                                       NGAINPN),
nGainAPD(                                                                     NGAINAPD)

//========================================================================

{

  // Initialization from cfg file

  resdir_                 = iConfig.getUntrackedParameter<std::string>("resDir");

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

  if(  _type!=  "TESTPULSE" ){
    cout << "Error: Wrong type "<<_type<<", should be LASER or LED or TESTPULSE"<< endl;
    cout << " Setting type to TESTPULSE "<<endl;
    _type="TESTPULSE";
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
EcalTestPulseAnalyzer::~EcalTestPulseAnalyzer(){
  //========================================================================


  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)

}

//========================================================================
void EcalTestPulseAnalyzer::beginJob() {
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
    ADCTrees[i]->SetBranchAddress( "adc",         adc          );
    ADCTrees[i]->SetBranchAddress( "adcGain",     &adcGain     );
    ADCTrees[i]->SetBranchAddress( "pn0",         &pn0         );
    ADCTrees[i]->SetBranchAddress( "pn1",         &pn1         ); 
    ADCTrees[i]->SetBranchAddress( "pnGain",      &pnGain      );    
 

  } 
 
  if ( _debug == 1 ) cout << "-- debug --  beginJob - _type = "<<_type<<" _typefit = "<<_typefit<< endl;
  
  // APD results file
  
  stringstream nameapdfile;
  nameapdfile << resdir_ <<"/APDPN_"<<_type<<".root";      
  resfile=nameapdfile.str();
  
  
  // Good type events counter
  
  typeEvents=0;
  
}

//========================================================================
void EcalTestPulseAnalyzer:: analyze( const edm::Event & e, const  edm::EventSetup& c){
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

    // Check fed corresponds to the DCC in TCC
    
    if( 600+dccID != fedID ) continue;
    
    // Cut on runType
    
    if(_type == "TESTPULSE"
       && runType!=EcalDCCHeaderBlock::TESTPULSE_MGPA 
       && runType!=EcalDCCHeaderBlock::TESTPULSE_GAP 
       && runType!=EcalDCCHeaderBlock::TESTPULSE_SCAN_MEM ) return;
    
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
    
    // Fill PN ampl vector

    allPNAmpl[pnDetId.iDCCId()].push_back(pnAmpl);
      
    if ( _debug == 2 ) cout <<"-- debug -- analyze: in PNDigi - PNampl=" << 
		     pnAmpl<<", PNgain="<< pnGain<<endl;  
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
           
      std::pair<int,int> pnpair=MEEBGeom::pn(module,_fedid);
      unsigned int MyPn0=pnpair.first;
      unsigned int MyPn1=pnpair.second;
      
      int lmr=MEEBGeom::lmr( etaG,phiG ); 
      unsigned int channel=MEEBGeom::electronic_channel( etaL, phiL );

      assert( channel <= nCrys );
      
      side=MEEBGeom::side(etaG,phiG);

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
      
      if(adcGain>3 || adcGain <0 ){
	nEvtBadGain[channel]++;   
	cout<<"BAD GAIN: "<<adcGain<< endl;
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
      
      if(adcGain>3 || adcGain <0 ){
	nEvtBadGain[channel]++;   
	cout<<"BAD GAIN: "<<adcGain<< endl;
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
	
      }
    }
  }
}

void EcalTestPulseAnalyzer::endJob() {
  
  
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

	unsigned int iM=modules[iMod]-1;

	TPPNAnal[iM][ich][ig]=new TPN(ich);
      }
    }
  }

  TSFit * pstpfit = new TSFit(_nsamples,650);
  pstpfit -> set_params(_nsamples, _niter, _presample, _samplemin, _samplemax, _timeofmax ,  _chi2max, _firstsample,  _lastsample);


  if ( _debug == 1 ) cout << "-- debug -- ncrys = "<< nCrys<<
    " nGainAPD = "<<nGainAPD<< " nMod="<< nMod<<endl;

  for (unsigned int iCry=0;iCry<nCrys;iCry++){

    for(unsigned int iG=0;iG<nGainAPD;iG++){
      TPAPDAnal[iCry][iG]=new TAPD();
    }

    
    if ( _debug == 1 ) cout << "-- debug --  moduleID="<<moduleID<<" nPNPerMod="<<nPNPerMod<< endl;

    Long64_t nbytes = 0, nb = 0;
    for (Long64_t jentry=0; jentry<  ADCTrees[iCry]->GetEntriesFast();jentry++) { 
      nb = ADCTrees[iCry]->GetEntry(jentry);   nbytes += nb; 
      
      pstpfit -> init_errmat(10.);
      
      unsigned int iMod;
      if(_ecalPart=="EB") iMod=MEEBGeom::lmmod(eta, phi)-1;
      else{
	iMod=MEEEGeom::lmmod((eta-1)/5+1,(phi-1)/5+1);
	if( iMod>=18 && side==1 ) iMod+=2;
	iMod--;
      }
      
      if(  isFirstChanModFilled[iMod]==0 ){ 
	firstChanMod[iMod]=iCry;
	isFirstChanModFilled[iMod]=1;
	if ( _debug == 4 ) cout <<"-- debug EcalLightAnalyzer -- endJobLight - filling firstChanMod :"<<iMod<<" "<< iCry<< endl;
      }
      
      moduleID=iMod+1;    
      if( moduleID>=20 ) moduleID-=2;  // Trick to fix endcap specificity
      
      if ( _debug == 1 ) cout << "-- debug -- before fit jentry="<< jentry<<" firstchanmod="<<firstChanMod[iMod]<<" icry="<<iCry<<" pnGain="<<pnGain<<" apdGain="<<adcGain<<" iMod="<< iMod<<endl;
      
      if ( _debug == 1 ) cout << "-- debug -- pn0="<< pn0<<" pn1="<<pn1<<endl;

      // PN Means and RMS 
      //==================

      if( firstChanMod[iMod] == iCry && isFirstChanModFilled[iMod]==1 ){ 
	for (unsigned int ichan=0;ichan<nPNPerMod;ichan++){
	  TPPNAnal[iMod][ichan][pnGain]->addEntry(pn0,pn1);
	}
      }
      

      if ( _debug == 1 ) cout << "-- debug -- adc[0]="<< adc[0]<<" adc[5]="<<adc[5]<<" adc[6]="<<adc[6]<< endl;
      
      // APD means and RMS
      //==================
      
      APDPulse->setPulse(adc); 
      if ( _debug == 1 ) cout << "-- debug -- setPulse OK"<< endl;
 
      adcNoPed=APDPulse->getAdcWithoutPedestal();
      
      if ( _debug == 1 ) cout << "-- debug --  isPulseOK="<< APDPulse->isPulseOK() <<" "<<pnGain<<" "<<adcGain<<endl;
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
	if ( _debug == 1 ) cout << "-- debug --  gain="<<adcGain<<" apdAmpl="<<apdAmpl<<endl;
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
      
      //if( APD[0]!=0 && APD[3]>10 ){ // Fill Anyway... What about a flag?
      if(gain==1) resTree->Fill(); 
	//}
    }
  }
  
  // Get final results for PN and PN/PN
  //====================================
  
  for (unsigned int ig=0;ig<nGainPN;ig++){
    for (unsigned int iMod=0;iMod<nMod;iMod++){
      for (int ch=0;ch<2;ch++){
	
	pnID=ch;
	moduleID=modules[iMod];
	if( moduleID>=20 ) moduleID-=2;  // Trick to fix endcap specificity
	std::vector<double> pnvec = TPPNAnal[modules[iMod]-1][ch][ig]->getPN();
	for(unsigned int i=0;i<pnvec.size();i++){
	  PN[i]=pnvec.at(i);
	}
	gain=ig;
	
	// Fill PN tree
	//==============
      
	//if( PN[0]!=0 && PN[3]>10 ){// Fill Anyway... What about a flag?
	if(gain==1) resPNTree->Fill();
	  //}
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


  
DEFINE_FWK_MODULE(EcalTestPulseAnalyzer);

