/* 
 *  \class EcalPerEvtLaserAnalyzer
 *
 *  $Date: 2012/02/09 10:07:37 $
 *  primary author: Julie Malcles - CEA/Saclay
 *  author: Gautier Hamel De Monchenault - CEA/Saclay
 */

#include <TFile.h>
#include <TTree.h>

#include "EcalPerEvtLaserAnalyzer.h"

#include <sstream>
#include <iostream>
#include <iomanip>
#include <ctime>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/MEEEGeom.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/MEEBGeom.h>

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>

#include <Geometry/EcalMapping/interface/EcalElectronicsMapping.h>
#include <Geometry/EcalMapping/interface/EcalMappingRcd.h>

#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalDetId/interface/EcalElectronicsId.h>
#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>

#include <vector>

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TPNFit.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/PulseFitWithFunction.h>

using namespace std;


//========================================================================
EcalPerEvtLaserAnalyzer::EcalPerEvtLaserAnalyzer(const edm::ParameterSet& iConfig)
  //========================================================================
  :
iEvent(0),

// framework parameters with default values
_nsamples(        iConfig.getUntrackedParameter< unsigned int >( "nSamples",           10 ) ),
_presample(       iConfig.getUntrackedParameter< unsigned int >( "nPresamples",         3 ) ),
_firstsample(     iConfig.getUntrackedParameter< unsigned int >( "firstSample",         1 ) ),
_lastsample(      iConfig.getUntrackedParameter< unsigned int >( "lastSample",          2 ) ),
_samplemin(       iConfig.getUntrackedParameter< unsigned int >( "sampleMin",           3 ) ),
_samplemax(       iConfig.getUntrackedParameter< unsigned int >( "sampleMax",           9 ) ),
_nsamplesPN(      iConfig.getUntrackedParameter< unsigned int >( "nSamplesPN",         50 ) ), 
_presamplePN(     iConfig.getUntrackedParameter< unsigned int >( "nPresamplesPN",       6 ) ),
_firstsamplePN(   iConfig.getUntrackedParameter< unsigned int >( "firstSamplePN",       7 ) ),
_lastsamplePN(    iConfig.getUntrackedParameter< unsigned int >( "lastSamplePN",        8 ) ),
_timingcutlow(    iConfig.getUntrackedParameter< unsigned int >( "timingCutLow",        3 ) ),
_timingcuthigh(   iConfig.getUntrackedParameter< unsigned int >( "timingCutHigh",       7 ) ),
_niter(           iConfig.getUntrackedParameter< unsigned int >( "nIter",               3 ) ),
_fedid(           iConfig.getUntrackedParameter< unsigned int >( "fedID",               0 ) ),
_tower(           iConfig.getUntrackedParameter< unsigned int >( "tower",               1 ) ),
_channel(         iConfig.getUntrackedParameter< unsigned int >( "channel",             1 ) ),
_ecalPart(        iConfig.getUntrackedParameter< std::string  >( "ecalPart",         "EB" ) ),
nCrys(                                                                               NCRYSEB),
nTT(                                                                                   NTTEB),
nSides(                                                                               NSIDES),
IsFileCreated(0), nCh(0),
runType(-1), runNum(0), dccID(-1), lightside(2), doesRefFileExist(0),
ttMat(-1), peakMat(-1), peak(-1), evtMat(-1), colMat(-1)
  //========================================================================

{

  //now do what ever initialization is needed

  resdir_                 = iConfig.getUntrackedParameter<std::string>("resDir");
  refalphabeta_           = iConfig.getUntrackedParameter<std::string>("refAlphaBeta");

  digiCollection_         = iConfig.getParameter<std::string>("digiCollection");
  digiPNCollection_       = iConfig.getParameter<std::string>("digiPNCollection");
  digiProducer_           = iConfig.getParameter<std::string>("digiProducer");

  eventHeaderCollection_  = iConfig.getParameter<std::string>("eventHeaderCollection");
  eventHeaderProducer_    = iConfig.getParameter<std::string>("eventHeaderProducer");

  // Define geometrical constants 

  if (_ecalPart == "EB") {
    nCrys     = NCRYSEB;
    nTT       = NTTEB;
  } else {
    nCrys     = NCRYSEE;
    nTT       = NTTEE;
  }

}

//========================================================================
EcalPerEvtLaserAnalyzer::~EcalPerEvtLaserAnalyzer(){
  //========================================================================


  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}



//========================================================================
void EcalPerEvtLaserAnalyzer::beginJob() {
  //========================================================================
 

  // Define temporary files' names

  stringstream namefile1;
  namefile1 <<resdir_<< "/ADCSamples.root" ;
      
  ADCfile=namefile1.str();
  
  // Create temporary file and trees to save adc samples
  
  ADCFile = new TFile(ADCfile.c_str(),"RECREATE");
  
    
  stringstream name;
  name <<"ADCTree";

  ADCtrees= new TTree(name.str().c_str(),name.str().c_str());  

  ADCtrees->Branch( "iphi",        &phi,   "phi/I"     );
  ADCtrees->Branch( "ieta",        &eta,   "eta/I"     );
  ADCtrees->Branch( "dccID",       &dccID, "dccID/I"   );
  ADCtrees->Branch( "event",       &event, "event/I"   );
  ADCtrees->Branch( "color",       &color, "color/I"   );
  ADCtrees->Branch( "adc",         &adc ,  "adc[10]/D" );
  ADCtrees->Branch( "ttrigMatacq", &ttrig, "ttrig/D"   );
  ADCtrees->Branch( "peakMatacq",  &peak,  "peak/D"    );
  ADCtrees->Branch( "pn0",         &pn0,   "pn0/D"     );
  ADCtrees->Branch( "pn1",         &pn1,   "pn1/D"     );
  
  ADCtrees->SetBranchAddress( "ieta",        &eta   );  
  ADCtrees->SetBranchAddress( "iphi",        &phi   ); 
  ADCtrees->SetBranchAddress( "dccID",       &dccID ); 
  ADCtrees->SetBranchAddress( "event",       &event );   
  ADCtrees->SetBranchAddress( "color",       &color );   
  ADCtrees->SetBranchAddress( "adc",         adc    ); 
  ADCtrees->SetBranchAddress( "ttrigMatacq", &ttrig );
  ADCtrees->SetBranchAddress( "peakMatacq",  &peak  );
  ADCtrees->SetBranchAddress( "pn0",         &pn0   );
  ADCtrees->SetBranchAddress( "pn1",         &pn1   );

  IsFileCreated=0;
  nCh=nCrys;
  
}


//========================================================================
void EcalPerEvtLaserAnalyzer:: analyze( const edm::Event & e, const  edm::EventSetup& c){
  //========================================================================

  ++iEvent;

  // retrieving DCC header
  edm::Handle<EcalRawDataCollection> pDCCHeader;
  const  EcalRawDataCollection* DCCHeader=0;
  try {
    e.getByLabel(eventHeaderProducer_,eventHeaderCollection_, pDCCHeader);
    DCCHeader=pDCCHeader.product();
  }catch ( std::exception& ex ) {
    std::cerr << "Error! can't get the product  retrieving DCC header" << eventHeaderCollection_.c_str() << std::endl;
  }
 
  // retrieving crystal data from Event
  edm::Handle<EBDigiCollection>  pEBDigi;
  const  EBDigiCollection* EBDigi=0;
  edm::Handle<EEDigiCollection>  pEEDigi;
  const  EEDigiCollection* EEDigi=0;

  if (_ecalPart == "EB") {
    try {
      e.getByLabel(digiProducer_,digiCollection_, pEBDigi); 
      EBDigi=pEBDigi.product(); 
    }catch ( std::exception& ex ) {
      std::cerr << "Error! can't get the product retrieving EB crystal data " << digiCollection_.c_str() << std::endl;
    } 
  } else {
    try {
      e.getByLabel(digiProducer_,digiCollection_, pEEDigi); 
      EEDigi=pEEDigi.product(); 
    }catch ( std::exception& ex ) {
      std::cerr << "Error! can't get the product retrieving EE crystal data " << digiCollection_.c_str() << std::endl;
    } 
  }
  
  
  // retrieving crystal PN diodes from Event
  edm::Handle<EcalPnDiodeDigiCollection>  pPNDigi;
  const  EcalPnDiodeDigiCollection* PNDigi=0;
  try {
    e.getByLabel(digiProducer_,digiPNCollection_, pPNDigi);
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

  // ====================================
  // Decode Basic DCCHeader Information 
  // ====================================

  for ( EcalRawDataCollection::const_iterator headerItr= DCCHeader->begin();headerItr != DCCHeader->end(); 
	++headerItr ) {
    
    // Get run type and run number 
    
    int fed = headerItr->fedId();      

    if(fed!=_fedid && _fedid!=-999) continue; 

    runType=headerItr->getRunType();
    runNum=headerItr->getRunNumber();
    event=headerItr->getLV1();
    dccID=headerItr->getDccInTCCCommand();
    fedID=headerItr->fedId();  

    // take event only if the fed corresponds to the DCC in TCC
    if( 600+dccID != fedID ) continue;
    
    // Cut on runType
    if(runType!=EcalDCCHeaderBlock::LASER_STD && runType!=EcalDCCHeaderBlock::LASER_GAP) return; 

    // Define output results files' names

    if (IsFileCreated==0){
 
      stringstream namefile2;
      namefile2 << resdir_ <<"/APDAmpl_Run"<<runNum<<"_"<<_fedid <<"_"<<_tower<<"_"<<_channel<<".root";      
      resfile=namefile2.str();
      
      // Get Matacq ttrig
      
      stringstream namefile;
      namefile << resdir_ <<"/Matacq-Run"<<runNum<<".root";      

      doesRefFileExist=0;

      FILE *test; 
      test = fopen(namefile.str().c_str(),"r");
      if (test) doesRefFileExist=1;
      
      if(doesRefFileExist==1){ 
	matacqFile = new TFile((namefile.str().c_str()));	  
	matacqTree=(TTree*)matacqFile->Get("MatacqShape");  

	matacqTree->SetBranchAddress( "event",       &evtMat  );
	matacqTree->SetBranchAddress( "color",       &colMat  );
	matacqTree->SetBranchAddress( "peak",        &peakMat );
	matacqTree->SetBranchAddress( "ttrig",       &ttMat   );
      }
      
      IsFileCreated=1;
      
    }
    
    // Retrieve laser color and event number

    EcalDCCHeaderBlock::EcalDCCEventSettings settings = headerItr->getEventSettings();
    int color = settings.wavelength;

    vector<int>::iterator iter = find( colors.begin(), colors.end(), color );
    if( iter==colors.end() ){
      colors.push_back( color );
      cout <<" new color found "<< color<<" "<< colors.size()<< endl;
    }

  }

  // cut on fedID

  if(fedID!=_fedid && _fedid!=-999) return; 

  
  // ======================
  // Decode PN Information
  // ======================
  
  TPNFit * pnfit = new TPNFit();
  pnfit -> init(_nsamplesPN,_firstsamplePN,  _lastsamplePN);
  
  double chi2pn=0;
  double ypnrange[50];
  double dsum=0.;
  double dsum1=0.;
  double bl=0.;
  double bl1=0.;
  double val_max=0.;
  unsigned int samplemax=0;
  unsigned int k;
  
  std::vector<double> allPNAmpl;
  
  for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNDigi->begin(); pnItr != PNDigi->end(); ++pnItr ) { // Loop on PNs 
    
    
    for ( int samId=0; samId < (*pnItr).size() ; samId++ ) { // Loop on PN samples  
      pn[samId]=(*pnItr).sample(samId).adc(); 
    }
    
    
    for(dsum=0.,k=0;k<_presamplePN;k++) {
      dsum+=pn[k]; 							  
    }
    bl=dsum/((double) _presamplePN);
    
    for(val_max=0.,k=0;k<_nsamplesPN;k++) {
      ypnrange[k]=pn[k] - bl;
      
      if(ypnrange[k] > val_max) {
	val_max= ypnrange[k]; samplemax=k; 
      } 
    }
    
    chi2pn = pnfit -> doFit(samplemax,&ypnrange[0]); 
    
    if(chi2pn == 101 || chi2pn == 102 || chi2pn == 103) pnAmpl=0.;
    else pnAmpl= pnfit -> getAmpl();

    allPNAmpl.push_back(pnAmpl);    

  }

  // ===========
  // Get Matacq
  // ===========

  ttrig=-1.;
  peak=-1.;
  
  if(doesRefFileExist==1){ 
    // FIXME 
    if (color==0) matacqTree->GetEntry(event-1);
    else if(color==3) matacqTree->GetEntry(matacqTree->GetEntries("color==0")+event-1);
    ttrig=ttMat;
    peak=peakMat;
    
  }
  
  
  // ===========================
  // Decode EBDigis Information
  // ===========================

  double yrange[10]; 
  int  adcGain=0;
  int side=0;

  if (EBDigi){
    
    for ( EBDigiCollection::const_iterator digiItr= EBDigi->begin(); digiItr != EBDigi->end(); ++digiItr ) {  // Loop on crystals
      
      EBDetId id_crystal(digiItr->id()) ;
      EBDataFrame df( *digiItr );

      int etaG = id_crystal.ieta() ;  // global
      int phiG = id_crystal.iphi() ;  // global

      int etaL ;  // local
      int phiL ;  // local
      std::pair<int, int> LocalCoord=MEEBGeom::localCoord( etaG , phiG );
      
      etaL=LocalCoord.first ;
      phiL=LocalCoord.second ;
      
      eta = etaG;
      phi = phiG;

      side=MEEBGeom::side(etaG,phiG);

      EcalElectronicsId elecid_crystal = TheMapping->getElectronicsId(id_crystal);

      int towerID=elecid_crystal.towerId();
      // int channelID=elecid_crystal.channelId()-1;  // FIXME so far for endcap only    
      int strip=elecid_crystal.stripId();
      int xtal=elecid_crystal.xtalId();
      int channelID=  5*(strip-1) + xtal-1; // FIXME

      int module= MEEBGeom::lmmod(etaG, phiG);

      std::pair<int,int> pnpair=MEEBGeom::pn(module);
      unsigned int MyPn0=pnpair.first;
      unsigned int MyPn1=pnpair.second;
                
      unsigned int channel=MEEBGeom::electronic_channel( etaL, phiL );
      assert(channel < nCrys); 
      
      double adcmax=0.0;
      
      if(towerID!=int(_tower) || channelID!=int(_channel) || dccID!=int(_fedid-600)) continue;
      else channelNumber=channel;
      
      for (unsigned int i=0; i< (*digiItr).size() ; ++i ) {  // Loop on adc samples  

	EcalMGPASample samp_crystal(df.sample(i));
	adc[i]=samp_crystal.adc() ;    
	adcG[i]=samp_crystal.gainId();   
	
	if (i==0) adcGain=adcG[i];
	if (i>0) adcGain=TMath::Max(adcG[i],adcGain);  
	
	if (adc[i]>adcmax) {
	  adcmax=adc[i];
	}
      }
      
      for(dsum=0.,dsum1=0.,k=0;k<_presample;k++) {
	dsum+=adc[k]; 
	if(k<_presample-1) dsum1+=adc[k];
      }
      
      bl=dsum/((double)_presample);
      bl1=dsum1/((double)_presample-1);
      
      for(val_max=0.,k=0;k<_nsamples;k++) {
	yrange[k]=adc[k] - bl;
	if(yrange[k] > val_max) {
	  val_max= yrange[k]; samplemax=k;
	}
      } 
      
      if(samplemax==4 || samplemax==5) {
	val_max=val_max+bl-bl1;
	for(k=0;k<_nsamples;k++) {
	  yrange[k]=yrange[k]+bl-bl1;
	}
      }

      for(unsigned int k=0;k<_nsamples;k++) { 
	adc[k]=yrange[k];
      }  
      
      pn0=allPNAmpl[MyPn0];
      pn1=allPNAmpl[MyPn1];
    
      if(samplemax>=_timingcutlow && samplemax<=_timingcuthigh && lightside==side)  ADCtrees->Fill();  
      
    }
 
  } else {
   
    for ( EEDigiCollection::const_iterator digiItr= EEDigi->begin(); digiItr != EEDigi->end(); ++digiItr ) {  // Loop on crystals
      
      EEDetId id_crystal(digiItr->id()) ;
      EEDataFrame df( *digiItr );

      phi = id_crystal.ix()-1 ; 
      eta = id_crystal.iy()-1 ; 
      side=0; // FIXME

      // Recover the TT id and the electronic crystal numbering from EcalElectronicsMapping

      EcalElectronicsId elecid_crystal = TheMapping->getElectronicsId(id_crystal);
    
      int towerID=elecid_crystal.towerId();
      int channelID=elecid_crystal.channelId()-1;  
    
      int module=MEEEGeom::lmmod( phi, eta );
      
      std::pair<int,int> pnpair=MEEEGeom::pn( module, _fedid ) ; 
      unsigned int MyPn0=pnpair.first;
      unsigned int MyPn1=pnpair.second;

      unsigned int channel=MEEEGeom::crystal( phi, eta ); 
      assert(channel < nCrys); 

      double adcmax=0.0;

      if(towerID!=int(_tower) || channelID!=int(_channel) || dccID!=int(_fedid-600)) continue;
      else channelNumber=channel;

      for (unsigned int i=0; i< (*digiItr).size() ; ++i ) {  // Loop on adc samples  

	EcalMGPASample samp_crystal(df.sample(i));
	adc[i]=samp_crystal.adc() ;    
	adcG[i]=samp_crystal.gainId();   
	
	if (i==0) adcGain=adcG[i];
	if (i>0) adcGain=TMath::Max(adcG[i],adcGain);  
	
	if (adc[i]>adcmax) {
	  adcmax=adc[i];
	}   
      }

      for(dsum=0.,dsum1=0.,k=0;k<_presample;k++) {
	dsum+=adc[k]; 
	if(k<_presample-1) dsum1+=adc[k];
      }
      
      bl=dsum/((double)_presample);
      bl1=dsum1/((double)_presample-1);
      
      for(val_max=0.,k=0;k<_nsamples;k++) {
	yrange[k]=adc[k] - bl;
	if(yrange[k] > val_max) {
	  val_max= yrange[k]; samplemax=k;
	}
      } 
    
      if(samplemax==4 || samplemax==5) {
	val_max=val_max+bl-bl1;
	for(k=0;k<_nsamples;k++) {
	  yrange[k]=yrange[k]+bl-bl1;
	}
      }
    
      for(unsigned int k=0;k<_nsamples;k++) { 
	adc[k]=yrange[k];
      }

      pn0=allPNAmpl[MyPn0];
      pn1=allPNAmpl[MyPn1];

      if(samplemax>=_timingcutlow && samplemax<=_timingcuthigh && lightside==side)  ADCtrees->Fill();  
    
    }
  }

}// analyze


//========================================================================
void EcalPerEvtLaserAnalyzer::endJob() {
//========================================================================


  assert( colors.size()<= nColor );
  unsigned int nCol=colors.size();


  ADCtrees->Write();
  
  
  cout <<  "\n\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;
  cout <<    "\t+=+       Analyzing laser data: getting per event               +=+" << endl;
  cout <<    "\t+=+            APD Amplitudes and ADC samples                   +=+"<< endl;
  cout <<    "\t+=+    for fed:" <<_fedid<<", tower:"<<_tower<<", and channel:" << _channel << endl;

    
  // Define temporary tree to save APD amplitudes

  APDFile = new TFile(resfile.c_str(),"RECREATE");
  
  int ieta, iphi, channelID, towerID, flag;
  double alpha, beta;
  

  colors.push_back( color );

  for (unsigned int i=0;i<nCol;i++){
    
    stringstream name1;
    name1<<"headerCol"<<colors[i];
    
    header[i] = new TTree(name1.str().c_str(),name1.str().c_str());
    
    header[i]->Branch( "alpha",     &alpha,     "alpha/D"     );
    header[i]->Branch( "beta",      &beta,      "beta/D"      );
    header[i]->Branch( "iphi",      &iphi,      "iphi/I"      );
    header[i]->Branch( "ieta",      &ieta,      "ieta/I"      );
    header[i]->Branch( "dccID",     &dccID,     "dccID/I"     );
    header[i]->Branch( "towerID",   &towerID,   "towerID/I"   );
    header[i]->Branch( "channelID", &channelID, "channelID/I" );
    
    header[i]->SetBranchAddress( "alpha",     &alpha     );
    header[i]->SetBranchAddress( "beta",      &beta      );
    header[i]->SetBranchAddress( "iphi",      &iphi      );
    header[i]->SetBranchAddress( "ieta",      &ieta      );
    header[i]->SetBranchAddress( "dccID",     &dccID     );
    header[i]->SetBranchAddress( "towerID",   &towerID   );
    header[i]->SetBranchAddress( "channelID", &channelID ); 
    
  }

  stringstream name2;
  name2<<"APDTree";  
  APDtrees= new TTree(name2.str().c_str(),name2.str().c_str());
    
  //List of branches
  
  APDtrees->Branch( "event",       &event,     "event/I"     );
  APDtrees->Branch( "color",       &color,     "color/I"     );
  APDtrees->Branch( "adc",         &adc ,      "adc[10]/D"   );
  APDtrees->Branch( "peakMatacq",  &peak ,     "peak/D"      );
  APDtrees->Branch( "ttrigMatacq", &ttrig ,    "ttrig/D"     );
  APDtrees->Branch( "apdAmpl",     &apdAmpl,   "apdAmpl/D"   );
  APDtrees->Branch( "apdTime",     &apdTime,   "apdTime/D"   );
  APDtrees->Branch( "flag",        &flag,      "flag/I"      ); 
  APDtrees->Branch( "pn0",         &pn0,       "pn0/D"       );
  APDtrees->Branch( "pn1",         &pn1,       "pn1/D"       ); 
  
  APDtrees->SetBranchAddress( "event",       &event     );
  APDtrees->SetBranchAddress( "color",       &color     );
  APDtrees->SetBranchAddress( "adc",         adc        );
  APDtrees->SetBranchAddress( "peakMatacq",  &peak      );
  APDtrees->SetBranchAddress( "ttrigMatacq", &ttrig     );
  APDtrees->SetBranchAddress( "apdAmpl",     &apdAmpl   );
  APDtrees->SetBranchAddress( "apdTime",     &apdTime   );
  APDtrees->SetBranchAddress( "flag",        &flag      ); 
  APDtrees->SetBranchAddress( "pn0",         &pn0       );
  APDtrees->SetBranchAddress( "pn1",         &pn1       ); 
  
  // Retrieve alpha and beta for APD amplitudes calculation
  

  TFile *alphaFile = new TFile(refalphabeta_.c_str());
  TTree *alphaTree[2];

  Double_t alphaRun, betaRun;
  int ietaRun, iphiRun, channelIDRun, towerIDRun, dccIDRun, flagRun;

  for( unsigned int i=0;i<nCol;i++){
    
    stringstream name3;
    name3<<"ABCol"<<i;  
    alphaTree[i]=(TTree*)alphaFile->Get(name3.str().c_str());
    alphaTree[i]->SetBranchStatus( "*",         0 );
    alphaTree[i]->SetBranchStatus( "alpha",     1 );
    alphaTree[i]->SetBranchStatus( "beta",      1 );
    alphaTree[i]->SetBranchStatus( "iphi",      1 );
    alphaTree[i]->SetBranchStatus( "ieta",      1 );
    alphaTree[i]->SetBranchStatus( "dccID",     1 );
    alphaTree[i]->SetBranchStatus( "towerID",   1 );
    alphaTree[i]->SetBranchStatus( "channelID", 1 );
    alphaTree[i]->SetBranchStatus( "flag",      1 );
       
    alphaTree[i]->SetBranchAddress( "alpha",     &alphaRun     );
    alphaTree[i]->SetBranchAddress( "beta",      &betaRun      );
    alphaTree[i]->SetBranchAddress( "iphi",      &iphiRun      );
    alphaTree[i]->SetBranchAddress( "ieta",      &ietaRun      );
    alphaTree[i]->SetBranchAddress( "dccID",     &dccIDRun     );
    alphaTree[i]->SetBranchAddress( "towerID",   &towerIDRun   );
    alphaTree[i]->SetBranchAddress( "channelID", &channelIDRun );
    alphaTree[i]->SetBranchAddress( "flag",      &flagRun      );
    
  }

  PulseFitWithFunction * pslsfit = new PulseFitWithFunction();
  
  double chi2;

  for (unsigned int icol=0;icol<nCol;icol++){
    
    IsThereDataADC[icol]=1;      
    stringstream cut;
    cut <<"color=="<<colors.at(icol);
    if(ADCtrees->GetEntries(cut.str().c_str())<10) IsThereDataADC[icol]=0;  
    IsHeaderFilled[icol]=0;

  }
    
    // Define submodule and channel number inside the submodule (as Patrice)
        

    Long64_t nbytes = 0, nb = 0;
    for (Long64_t jentry=0; jentry< ADCtrees->GetEntriesFast();jentry++) {  // Loop on events
      nb = ADCtrees->GetEntry(jentry);   nbytes += nb; 	
      
      int iCry=channelNumber;
      
      // get back color
      
      unsigned int iCol=0;
      for(unsigned int i=0;i<nCol;i++){
	if(color==colors[i]) {
	  iCol=i;
	  i=colors.size();
	}
      }

      alphaTree[iCol]->GetEntry(iCry);      
      

      flag=flagRun;
      iphi=iphiRun;
      ieta=ietaRun;
      towerID=towerIDRun;
      channelID=channelIDRun;
      alpha=alphaRun;
      beta=betaRun;

      if (IsHeaderFilled[iCol]==0){
	header[iCol]->Fill();
	IsHeaderFilled[iCol]=1;
	
      }
       // Amplitude calculation

      apdAmpl=0;
      apdTime=0;
      
      pslsfit -> init(_nsamples,_firstsample,_lastsample,_niter,  alphaRun,  betaRun);
      chi2 = pslsfit -> doFit(&adc[0]);
      
      if( chi2 < 0. || chi2 == 102 || chi2 == 101 ) {	  

	apdAmpl=0;	  
	apdTime=0;
	
      }else{
	
	apdAmpl = pslsfit -> getAmpl();
	apdTime = pslsfit -> getTime();
       
      }

      APDtrees->Fill();
      
    }

    alphaFile->Close();
    
    ADCFile->Close();

    APDFile->Write();
    APDFile->Close();


  

  // Remove unwanted files
 
  stringstream del;
  del << "rm " <<ADCfile;
  system(del.str().c_str()); 
  
  cout <<    "\t+=+    .................................................. done  +=+" << endl;
  cout <<    "\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;
}



DEFINE_FWK_MODULE(EcalPerEvtLaserAnalyzer);

