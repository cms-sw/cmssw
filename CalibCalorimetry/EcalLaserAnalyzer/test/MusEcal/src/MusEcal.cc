#include <iostream>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "MusEcal.hh"
#include "../../interface/MEGeom.h"
#include "../../interface/MEChannel.h"
#include "../../interface/TCalibData.h"
#include "MERunManager.hh"
#include "MERun.hh"
#include "MEVarVector.hh"
#include "MECanvasHolder.hh"
#include "MEIntervals.hh"
#include "METimeInterval.hh"
#include "MECorrector2Var.hh"
#include "../../interface/TPNCor.h"

#include <TMath.h>
#include <TGraph.h>
#include <TF1.h>

ClassImp(MusEcal)

bool MusEcal::verbose= true;

//int MusEcal::corVar = ME::iMTQ_FWHM;

TString MusEcal::historyVarTitle[MusEcal::iSizeLV] = {  
  "CLS" ,  // FIXME: put better names here namely "Glued Normalized LASER Signal", "Normalized LASER Signal", 
  "RabCLS" ,
  "NLS",
  "RabNLS", 
  "CLSNORM",
  "NLSNORM",
  "Ratio APD over PN corrected from LASER",
  "Ratio APD over PNA corrected from LASER",
  "Ratio APD over PNB corrected from LASER",
  "Ratio APD over PNA", 
  "Ratio APD over PNB", 
  "Ratio APD over PN",
  "Ratio APD over PNA corrected from PN linearity", 
  "Ratio APD over PNB  corrected from PN linearity", 
  "Ratio APD over PN  corrected from PN linearity",
  "Ratio APD ABFIX over PNA corrected from PN linearity ", 
  "Ratio APD ABFIX over PNB corrected from PN linearity ", 
  "Ratio APD ABFIX over PN  corrected  from PN linearity ", 
  "Ratio APD ABFIT over PNA corrected from PN linearity ", 
  "Ratio APD ABFIT over PNB corrected from PN linearity ", 
  "Ratio APD ABFIT over PN  corrected  from PN linearity ", 
  "Nevt APD over PNA",    
  "Nevt APD over PNB",     
  "Ratio APD over APDA",  
  "Ratio APD over APDB",  
  "Nevt APD over APDA",   
  "Nevt APD over APDB",   
  "APD",
  "APD Time",
  "Nevt APD",             
  "Nevt APD Time",        
  "Normalization PNA",
  "Normalization PNB",  
  "Nevt PNA",             
  "Nevt PNB",           
  "RMS PNA",             
  "RMS PNB",             
  "Ratio PNB over PNA",  
  "Shape Correction for PNA",
  "Shape Correction for PNB", 
  "Time Rise = alpha*beta",
  "Shape Correction for APD",
  "Shape Correction Ratio",
  "MATACQ Time Rise", 
  "MATACQ Amplitude", 
  "MATACQ Full Width at Half Maximum", 
  "MATACQ Full Width at 10\%", 
  "MATACQ Full Width at  5\%", 
  "MATACQ Time" 
};

TString MusEcal::historyVarName[MusEcal::iSizeLV] = { 
  "CLS" , 
  "RabCLS" ,
  "NLS",
  "RabNLS",
  "CLSNORM",
  "NLSNORM",
  "MIDA", 
  "MIDB", 
  "MID", 
  "APDoPNA", 
  "APDoPNB", 
  "APDoPN",
  "APDoPNACOR", 
  "APDoPNBCOR", 
  "APDoPNCOR",
  "APDABFIXoPNACOR", 
  "APDABFIXoPNBCOR", 
  "APDABFIXoPNCOR",
  "APDABFIToPNACOR", 
  "APDABFIToPNBCOR", 
  "APDABFIToPNCOR",
  "APDoPNA_Nevt",     
  "APDoPNB_Nevt",   
  "APDoAPDA",        
  "APDoAPDB",        
  "APDoAPDA_Nevt",    
  "APDoAPDB_Nevt",    
  "APD",
  "APDTime",
  "APD_Nevt",         
  "APDTime_Nevt",     
  "PNA",
  "PNB",  
  "PNA_Nevt",         
  "PNB_Nevt",    
  "PNA_RMS",     
  "PNB_RMS",       
  "PNBoPNA",  
  "ShapeCorrectionPNA",
  "ShapeCorrectionPNB",
  "AlphaBeta",
  "ShapeCorrectionAPD",
  "ShapeCorrectionRatio",
  "MATACQ_TimeRise", 
  "MATACQ_Amplitude", 
  "MATACQ_FWHM", 
  "MATACQ_FW10", 
  "MATACQ_FW05", 
  "MATACQ_time" 
};

int MusEcal::iGVar[MusEcal::iSizeLV] = {
  
  ME::iCrystal,  // NLS
  ME::iCrystal,  // RabNLS
  ME::iCrystal,  // NLS
  ME::iCrystal,  // RabNLS
  ME::iCrystal,  // NLS
  ME::iCrystal,  // RabNLS
  ME::iCrystal,  // MIDA
  ME::iCrystal,  // MIDB
  ME::iCrystal,  // MID
  ME::iCrystal,  // APDoPNA
  ME::iCrystal,  // APDoPNB
  ME::iCrystal,  // APDoPN
  ME::iCrystal,  // APDoPNACOR
  ME::iCrystal,  // APDoPNBCOR
  ME::iCrystal,  // APDoPNCOR
  ME::iCrystal,  // APDABFIToPNACOR
  ME::iCrystal,  // APDABFIToPNBCOR
  ME::iCrystal,  // APDABFIToPNCOR
  ME::iCrystal,  // APDABFIXoPNACOR
  ME::iCrystal,  // APDABFIXoPNBCOR
  ME::iCrystal,  // APDABFIXoPNCOR
  ME::iCrystal,  // APDoPNANevt
  ME::iCrystal,  // APDoPNBNevt
  ME::iCrystal,  // APDoAPDA      
  ME::iCrystal,  // APDoAPDB      
  ME::iCrystal,  // APDoAPDANevt  
  ME::iCrystal,  // APDoAPDBNevt  
  ME::iCrystal,  // APD
  ME::iCrystal,  // APDTime
  ME::iCrystal,  // APDNevt       
  ME::iCrystal,  // APDTimeNevt   
  ME::iLMModule, // PNA
  ME::iLMModule, // PNB
  ME::iLMModule, // PNANevt       
  ME::iLMModule, // PNBNevt    
  ME::iLMModule, // PNARMS       
  ME::iLMModule, // PNBRMS       
  ME::iLMModule, // PNBoPNA
  ME::iLMModule, // shape correction PNA
  ME::iLMModule, // shape correction PNB
  ME::iCrystal,  // alpha*beta
  ME::iCrystal,  // shape correction APD
  ME::iCrystal,  // shape correction Ratio
  ME::iLMRegion, // MATACQ time rise
  ME::iLMRegion, // MATACQ amplitude
  ME::iLMRegion, // MATACQ fwhm
  ME::iLMRegion, // MATACQ fw20
  ME::iLMRegion, // MATACQ fw80
  ME::iLMRegion  // MATACQ time
};

int MusEcal::historyVarZoom[ME::iSizeC][MusEcal::iSizeLV] = {
  { 
    MusEcal::iPercent,       // NLS
    MusEcal::iPercent,       // RabNLS
    MusEcal::iPercent,       // NLS
    MusEcal::iPercent,       // NLS
    MusEcal::iPercent,       // RabNLS
    MusEcal::iPercent,       // RabNLS
    MusEcal::iPercent,       // MIDA
    MusEcal::iPercent,       // MIDB
    MusEcal::iPercent,       // MID
    MusEcal::iZero,          // APDoPNA
    MusEcal::iZero,          // APDoPNB
    MusEcal::iPercent,       // APDoPN
    MusEcal::iZero,          // APDoPNACOR
    MusEcal::iZero,          // APDoPNBCOR
    MusEcal::iZero,          // APDoPNCOR
    MusEcal::iZero,          // APDABFIToPNACOR
    MusEcal::iZero,          // APDABFIToPNBCOR
    MusEcal::iZero,          // APDABFIToPNCOR
    MusEcal::iZero,          // APDABFIXoPNACOR
    MusEcal::iZero,          // APDABFIXoPNBCOR
    MusEcal::iZero,          // APDABFIXoPNCOR
    MusEcal::iZero,          // APDoPNANevt  
    MusEcal::iZero,          // APDoPNBNevt  
    MusEcal::iZero,          // APDoAPDA     
    MusEcal::iZero,          // APDoAPDB     
    MusEcal::iZero,          // APDoAPDANevt 
    MusEcal::iZero,          // APDoAPDBNevt 
    MusEcal::iZero,          // APD
    MusEcal::iZero,          // APDTime
    MusEcal::iZero,          // APDNevt     
    MusEcal::iZero,          // APDTimeNevt 
    MusEcal::iZero,          // PNA
    MusEcal::iZero,          // PNB
    MusEcal::iZero,          // PNANevt     
    MusEcal::iZero,          // PNBNevt     
    MusEcal::iZero,          // PNARMS     
    MusEcal::iZero,          // PNBRMS     
    MusEcal::iZero,          // PNBoPNA
    MusEcal::iPercent,       // Shape correction PNA
    MusEcal::iPercent,       // Shape correction PNB
    MusEcal::iZero,          // AlphaBeta
    MusEcal::iPercent,      // Shape correction APD
    MusEcal::iPercent,      // Shape correction Ratio
    MusEcal::iZero,          // MTQ_Trise
    MusEcal::iZero,          // MTQ_Ampl 
    MusEcal::iThirtyPercent, // MTQ_Fwhm 
    MusEcal::iZero,          // MTQ_Fw20 
    MusEcal::iZero,          // MTQ_Fw80 
    MusEcal::iZero           // MTQ_time 
  },
  {  MusEcal::iPercent,       // NLS
    MusEcal::iPercent,       // RabNLS
    MusEcal::iPercent,       // NLS
    MusEcal::iPercent,       // NLS
    MusEcal::iPercent,       // RabNLS
    MusEcal::iPercent,       // RabNLS
    MusEcal::iPercent,       // MIDA
    MusEcal::iPercent,       // MIDB
    MusEcal::iPercent,       // MID
    MusEcal::iZero,          // APDoPNA
    MusEcal::iZero,          // APDoPNB
    MusEcal::iPercent,       // APDoPN
    MusEcal::iZero,          // APDoPNACOR
    MusEcal::iZero,          // APDoPNBCOR
    MusEcal::iZero,          // APDoPNCOR
    MusEcal::iZero,          // APDABFIToPNACOR
    MusEcal::iZero,          // APDABFIToPNBCOR
    MusEcal::iZero,          // APDABFIToPNCOR
    MusEcal::iZero,          // APDABFIXoPNACOR
    MusEcal::iZero,          // APDABFIXoPNBCOR
    MusEcal::iZero,          // APDABFIXoPNCOR
    MusEcal::iZero,          // APDoPNANevt  
    MusEcal::iZero,          // APDoPNBNevt  
    MusEcal::iZero,          // APDoAPDA     
    MusEcal::iZero,          // APDoAPDB     
    MusEcal::iZero,          // APDoAPDANevt 
    MusEcal::iZero,          // APDoAPDBNevt 
    MusEcal::iZero,          // APD
    MusEcal::iZero,          // APDTime
    MusEcal::iZero,          // APDNevt     
    MusEcal::iZero,          // APDTimeNevt 
    MusEcal::iZero,          // PNA
    MusEcal::iZero,          // PNB
    MusEcal::iZero,          // PNANevt     
    MusEcal::iZero,          // PNBNevt     
    MusEcal::iZero,          // PNARMS     
    MusEcal::iZero,          // PNBRMS     
    MusEcal::iZero,          // PNBoPNA
    MusEcal::iPercent,       // Shape correction PNA
    MusEcal::iPercent,       // Shape correction PNB
    MusEcal::iZero,          // AlphaBeta
    MusEcal::iPercent,      // Shape correction APD
    MusEcal::iPercent,      // Shape correction Ratio
    MusEcal::iZero,          // MTQ_Trise
    MusEcal::iZero,          // MTQ_Ampl 
    MusEcal::iThirtyPercent, // MTQ_Fwhm 
    MusEcal::iZero,          // MTQ_Fw20 
    MusEcal::iZero,          // MTQ_Fw80 
    MusEcal::iZero           // MTQ_time 
  }
};

int MusEcal::historyVarColor[MusEcal::iSizeLV] = {

  kBlue,       // NLS
  kRed,        // RabMID
  kBlue,       // NLS
  kRed,        // RabMID
  kBlue,       // NLS
  kRed,        // RabMID
  kBlue,       // MIDA
  kBlue,       // MIDB
  kBlue,       // MID
  kBlue,       // APDoPNA
  kBlue,       // APDoPNB
  kBlue,       // APDoPN
  kBlue,       // APDoPNACOR
  kBlue,       // APDoPNBCOR
  kBlue,       // APDoPNCOR
  kBlue,       // APDABFIToPNACOR
  kBlue,       // APDABFIToPNBCOR
  kBlue,       // APDABFIToPNCOR
  kBlue,       // APDABFIXoPNACOR
  kBlue,       // APDABFIXoPNBCOR
  kBlue,       // APDABFIXoPNCOR
  kBlue,       // APDoPNANevt  
  kBlue,       // APDoPNBNevt  
  kBlue,       // APDoAPDA     
  kBlue,       // APDoAPDB     
  kBlue,       // APDoAPDANevt 
  kBlue,       // APDoAPDBNevt 
  kBlue,       // APD          
  kBlue,       // APDTime 
  kBlue,       // APDNevt      
  kBlue,       // APDTimeNevt  
  kRed,        // PNA
  kRed,        // PNB
  kRed,        // PNANevt      
  kRed,        // PNBNevt      
  kRed,        // PNARMS      
  kRed,        // PNBRMS      
  kRed,        // PNBoPNA
  kMagenta,    // Shape correction PNA
  kMagenta,    // Shape correction PNB
  kMagenta,    // AlphaBeta
  kMagenta,    // Shape correction APD
  kMagenta,    // Shape correction Ratio
  kGreen,      // MTQ_Trise
  kGreen,      // MTQ_Ampl
  kGreen,      // MTQ_Fwhm
  kGreen,      // MTQ_Fw20
  kGreen,      // MTQ_Fw80
  kGreen       // MTQ_time
};
  
TString MusEcal::historyTPVarName[MusEcal::iSizeTPV] = { 
  "Test-Pulse APD gain 0", 
  "Test-Pulse APD gain 1", 
  "Test-Pulse APD gain 2", 
  "Test-Pulse PNA gain 0",
  "Test-Pulse PNA gain 1",
  "Test-Pulse PNB gain 0",
  "Test-Pulse PNB gain 1"
};
  
TString MusEcal::historyTPVarTitle[MusEcal::iSizeTPV] = { 
  "Test-Pulse APD for Gain 0", 
  "Test-Pulse APD for Gain 1", 
  "Test-Pulse APD for Gain 2", 
  "Test-Pulse PNA for Gain 0",
  "Test-Pulse PNA for Gain 1",
  "Test-Pulse PNB for Gain 0",
  "Test-Pulse PNB for Gain 1"
};

int MusEcal::iGTPVar[MusEcal::iSizeTPV] = {
  ME::iCrystal, // Test-Pulse APD gain 0 
  ME::iCrystal, // Test-Pulse APD gain 1 
  ME::iCrystal, // Test-Pulse APD gain 2 
  ME::iLMModule,  // Test-Pulse PNA gain 0  
  ME::iLMModule,  // Test-Pulse PNA gain 1 
  ME::iLMModule,  // Test-Pulse PNB gain 0 
  ME::iLMModule   // Test-Pulse PNB gain 1
};

int MusEcal::historyTPVarColor[MusEcal::iSizeTPV] = {
  kCyan,          
  kCyan,      
  kBlue,          
  kBlue,          
  kRed,           
  kRed,       
  kRed
};

int MusEcal::historyTPVarZoom[MusEcal::iSizeTPV] = {
  MusEcal::iThreePerMil, 
  MusEcal::iThreePerMil, 
  MusEcal::iThreePerMil, 
  MusEcal::iZero, 
  MusEcal::iZero, 
  MusEcal::iZero, 
  MusEcal::iZero
};

TString MusEcal::zoomName[MusEcal::iZero] =   {
  "Range +/- 100%", "Range +/- 50%", "Range +/- 30%", "Range +/- 10%", 
  "Range +/- 5%", "Range +/- 3%", "Range +/- 1%",
  "Range +/- 0.5%", "Range +/- 0.3%", "Range +/- 0.1%"
};
  
double MusEcal::zoomRange[MusEcal::iZero] =   {
  1., 0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001
};

int MusEcal::firstRun = 0;
int MusEcal::lastRun = 999999;
int MusEcal::LMR = 1;
bool MusEcal::fillNLS = false;

MusEcal::MusEcal( int type, int color )
{
  if( verbose )
    cout << "Welcome to MusEcal" << endl;

  // set type and color
  if( type == ME::iTestPulse ) color=0;
  setType( type, color );

  // no debug by default
  _debug=false;
  _debug2=false;
  
  // no GUI by default
  _isGUI = false;
      
  // histograms
  _histoBooked = false;
  _ebHistoBooked = false;
  _eeHistoBooked = false;
  _seq_t=0;

  // default MTQ variables (for interval making at Side granularity)
  _mtqVar0  =  -1;
  _mtqVar1  =  -1;
  _mtqLevel =  -1;
  setMtqVar( ME::iMTQ_FWHM,-1, MEIntervals::level2 );
  _tpVar0  =  -1;
  _tpVar1  =  -1;
  _tpLevel =  -1;
  setTPVar( ME::iTPAPD_MEAN,-1, MEIntervals::level5 );
  _pnVar0  =  -1;
  _pnVar1  =  -1;
  _pnLevel =  -1;
  setPNVar( ME::iPNA_OVER_PNB_MEAN,-1, MEIntervals::level5 );


  //  _corVar1      = iMID;
  //  _corZoom1     = MusEcal::iThreePercent;
  //   _corFitDegree = 1;
  //   _corZoom0     = MusEcal::iOneHundredPercent;
  //   _corVar0      = corVar ;
  //   _corY0        =  1;
  //   _corBeta.clear();
  //  if( _debug )
  //     cout << "before switch" << endl;
  //   switch( _corVar0 )
  //     {
  //     case ME::iMTQ_FWHM:
  //       _corX0        = 34;
  //       _corBeta.push_back( 0 );
  //       _corBeta.push_back( -0.0035 ); 
  //       break;
  //     case ME::iMTQ_RISE:
  //       _corX0        = 10.82;
  //       //      _corBeta.push_back( -0.00835105 );  
  //       _corBeta.push_back( 0 );
  //       _corBeta.push_back( 0 );  
  //       break;
  //     case ME::iAPD_SHAPE_COR:
  //       _corX0        = 0.795;
  //       _corZoom0     = MusEcal::iPercent;
  //       _corZoom1     = MusEcal::iPercent;
  //       _corBeta.push_back( 0. );
  //       _corBeta.push_back( 0.3 );  
  //       _corFitDegree = 1;
  //       break;
  //     default:
  //       abort();
  //     }
  _nbuf=NBUF;
  _nbad=NBAD;
  _nave=NAVE;
  
  // set default crystal: first crystal of the first monitoring region
  // leaf: selected first crystal by default

  _lmr = -1;

  map< TString, MERunManager* >::iterator it = _runMgr.begin();  
  for( ; it!=_runMgr.end(); ++it )
    {
      MERunManager* mgr_ = it->second;
      if( mgr_->size()==0 ) continue;
      int ilmr = mgr_->LMRegion();
      if(ilmr != LMR ) continue; // JM
      cout << " setLMRegion "<< ilmr<< endl;
      setLMRegion( ilmr );
      break;
      
    }

  assert( it!=_runMgr.end() );

  if( _debug )
    cout << "set LMR done" << endl;
  
  if(_debug) cout << " before setChannel "  << endl;
  
  //setChannel( ME::iCrystal, 0, 0, false );             // Originak setChannel
  
  int reg_(0); int sect_(0); int dcc_(0); int side_(0);  //JM
  ME::regionAndSector( LMR, reg_, sect_, dcc_, side_ );  // JM
  MEChannel *defLeaf =  ME::regTree( reg_ )->getDescendant( ME::iLMRegion, LMR )
    ->getFirstDescendant( ME::iCrystal );                // JM
  setChannel(defLeaf);
  
  // set default time
  setDefaultTime();
  if( _debug )
      cout << "setDefaultTime done" << endl;
    
  // set default variable
  setVar();
  if( _debug )
    cout << "setVar done" << endl;
  
}

MusEcal::~MusEcal() 
{
  cout <<" Inside MusEcal destructor "<< endl;
 
}

void
MusEcal::setType( int type, int color )
{
  _type = type;
  if( verbose )
    cout << "Current type is: " << ME::type[_type] << endl;

  if( _type==ME::iLaser )
    {
      // default color
      _color = color;
      if( verbose )
	cout << "Current color is: " << ME::color[_color] << endl;
    }
  
  // Barrel: Laser monitoring regions between 1 and 92
  for( unsigned int lmr=1; lmr<=92; lmr++ )
    {
      TString str_ =MusEcal::mgrName(lmr,_type,_color);
      if( _runMgr.count(str_)==0 )
	{
	  MERunManager* runMgr_ = new MERunManager( lmr, _type, _color );
	  if( runMgr_->size()==0 )
	    {
	      delete runMgr_;
	      continue;
	    }
	  _runMgr[str_]=runMgr_;
	}
    }
}

void
MusEcal::setLMRegion( int lmr )
{


  // Set LMR:
  //===========
  if( lmr==_lmr ) return;
  if(_debug) cout << "MusEcal::setLMRegion, before Mgr refresh "  << endl;
  if( curMgr()!=0 ) curMgr()->refresh();


  _lmr = lmr;
  cout << "Current Laser monitoring region is: " 
       << _lmr
       <<  " (" << ME::smName( _lmr ) << ")" << endl;
  
  // Update TCalibData
  //===================
  
  if(_calibData !=0 ) delete _calibData;
  
  string calibpath(getenv("MECALIB"));
  string alphapath(getenv("MEAB"));

  //cout<<"test calib: "<<calibpath<<" "<<alphapath<< endl;

  std::pair< int, int > dccSide=ME::dccAndSide( _lmr );
  _calibData=new TCalibData(dccSide.first, calibpath, alphapath );

  // Filling maps
  //=============
  
  if(_debug) cout << "MusEcal::setLMRegion, before fillMaps "  << endl;
  curMgr()->fillMaps();
  if(_debug) cout << "MusEcal::setLMRegion, before fillMIDMaps "  << endl;
  fillMIDMaps();  
  if(fillNLS){
    if(_debug) cout << "MusEcal::setLMRegion, before fillNLSMaps "  << endl;
    fillNLSMaps();
  }
}

//
// access to run managers
//
MERunManager* 
MusEcal::runMgr( int lmr, int type, int color )
{
  TString str_ = MusEcal::mgrName( lmr, type, color );
  if( _runMgr.count(str_)==0 )
    {
      //      cout << "Non existent Run manager" << endl;
      return 0;
    }
  else
    return _runMgr[str_];
}

void
MusEcal::setTime( ME::Time time )
{
  time_t t = time;
  if( verbose )
    cout << "Reference date: " << ctime(&t);
  map< TString, MERunManager* >::iterator it = _runMgr.begin();
  unsigned int currun_(0);
  unsigned int curlb_(0);
  for( ; it!=_runMgr.end(); ++it )
    {
      MERunManager* mgr_ = it->second;
      bool ok = mgr_->setCurrentRun( time );
      short int sign_(0);
      vector< ME::Time > dtvec;
      int nmaxlb=7;
      if( ok )
	{
	  MERun* run_ = mgr_->curRun();
	  if( currun_==0 ) 
	    {
	      currun_ = run_->run();
	      curlb_  = run_->lb();
	    }
	  if( run_->run()!=currun_ ) ok=false;
	  else if( run_->lb()<curlb_ || run_->lb()>curlb_+nmaxlb ) ok=false;
	  else
	    {
	      curlb_ = run_->lb();
	      ME::Time t_ = run_->time();
	      dtvec = ME::timeDiff( t_, time, sign_ );
	      if( dtvec[ME::iDay]!=0 )         ok=false;
	      else if( dtvec[ME::iHour]!=0 )   ok=false;
	      else if( dtvec[ME::iMinute]>30 ) ok=false;
	    }
	}
      if( !ok )
	{	  
	  mgr_->setNoCurrent();
	}
      if( verbose )
	{
	  int lmr_ =  mgr_->LMRegion();
	  cout << "LMR=" << lmr_ 
	       << " (" << ME::smName( lmr_ ) << ")";
	  if( ok )
	    {	    
	       cout 
		 << " --> " << mgr_->curRun()->header().rundir	    
		 << " d=" << dtvec[ME::iDay] << "/"
		 << " h=" << dtvec[ME::iHour] << "/"
		 << " m=" << dtvec[ME::iMinute] << "/"
		 << " s=" << dtvec[ME::iSecond];
	       if( sign_!=0 )
		 cout << ( (sign_>0)?" in future":" -------" );
	    }
	  else
	    {
	      cout << "--> not in sequence ";
	    }
	  cout << endl;
	}
    }
  //  _time = time;
  //  _time = _runMgr.begin()->second->curRun()->time();
  _time = runMgr()->curRun()->time();
}

bool
MusEcal::nextSequence()
{
  // fixme: this assumes that the current LMR is always present
  //  MERunManager* mgr_ = _runMgr.begin()->second;
  MERunManager* mgr_ = runMgr();
  MusEcal::RunIterator it = mgr_->cur();
  if( it==mgr_->end() ) return false;
  it++;
  if( it==mgr_->end() ) return false;
  ME::Time time    = it->first;
  setTime( time );
  return true;
}

void
MusEcal::setRunAndSequence( unsigned int irun, int iseq )
{
  MERunManager* mgr_ = _runMgr.begin()->second;
  // MERunManager* mgr_ = runMgr();
  MusEcal::RunIterator runit = mgr_->it();
  for( ; runit!=mgr_->end(); ++runit )
    {
      MERun* run_ = runit->second;
      if( run_->run()==irun ) iseq--;
      if( iseq==0 ) break;
    }
  ME::Time time = runit->second->time();
  setTime( time );
}

void
MusEcal::setDefaultTime()
{
  MERunManager* mgr_ = _runMgr.begin()->second;
  // MERunManager* mgr_ = runMgr();
  MERun* run_ = mgr_->firstRun();
  if( run_==0 ) 
    {
      cout << "run pointer is null " << endl;
      return;
    }
  ME::Time time = run_->time();
  setTime( time );
}

void
MusEcal::dumpVector( int ivar )
{
  // dummy !!!
  //  cout << "Dump current vector as ascii file" << endl;
  if( _leaf==0 ) 
    {
      cout << "Please select a channel first " << endl;
      return;
    }
  // test
  MEChannel* leaf_ = _leaf;
  MEVarVector* apdVector_ = curMgr()->apdVector(leaf_);
  vector< ME::Time > time;
  vector< float > val;
  vector< bool > flag;
  int ii = ivar;
  apdVector_->getTimeValAndFlag( ii, time, val, flag );
  unsigned int nrun = time.size();
  //  cout << endl;
  //  cout << leaf_->oneLine() << endl;
  for( unsigned int irun=0; irun<nrun; irun++ )
    {
      float dt_ = ME::timeDiff( time[irun], _time, ME::iHour );
      float val_ = val[irun];
      bool flag_ = flag[irun];
      TString str_ = ME::APDPrimVar[ii];
      time_t t = time[irun];
      struct tm *tb;
      tb = localtime(&t);
      //      cout << ctime(&t);
      int mon_ = tb->tm_mon+1;      
      int day_ = tb->tm_mday;      
      int hr_  = tb->tm_hour+1;
      int min_ = tb->tm_min;
      int sec_ = tb->tm_sec;      
      cout << tb->tm_year+1900 << "/";
      if( mon_<10 ) cout << "0";
      cout << mon_ << "/"; 
      if( day_<10 ) cout << "0";
      cout << day_ << "-";
      if( hr_<10 ) cout << "0";
      cout << hr_ << ":";
      if( min_<10 ) cout << "0";
      cout << min_ << ":";
      if( sec_<10 ) cout << "0";
      cout << sec_;
      cout << "\tt(sec)=" << t;
      cout << "\tdt(hrs)=" << dt_;
      cout << "\t\t" << str_ << "=" << val_ << "\tok=" << flag_ << endl;      
    }

}

void 
MusEcal::setChannel( int ig, int ix, int iy, bool useGlobal )
{
  int reg_(0); int sect_(0); int dcc_(0); int side_(0);
  ME::regionAndSector( _lmr, reg_, sect_, dcc_, side_ );
  MEChannel* leaf_(0);

  if( reg_==ME::iEBM || reg_==ME::iEBP )
    {	  
      if( !useGlobal )
	{
	  if( _lmr<0 ) _lmr=1;
	  MEEBGeom::EtaPhiCoord etaphi_ 
	    = MEEBGeom::globalCoord( sect_, ix, iy );
	  ix = etaphi_.first;
	  iy = etaphi_.second;
	}

      // FIXME: ONLY BARREL FOR THE MOMENT
      assert( ix!=0 && abs(ix)<=85 );
      assert( iy>=1 && iy<=360 );
      if( ix<0 ) reg_ = ME::iEBM;
      if( ix>0 ) reg_ = ME::iEBP;
      leaf_ = ME::regTree( reg_ )->getChannel( ig, ix, iy )  ;
    }
  else
    {
      if( !useGlobal )
	{
	  leaf_ =  
	    ME::regTree( reg_ )->getDescendant( ME::iLMRegion, _lmr )
	    ->getFirstDescendant( ME::iCrystal );
	  
	}
      else
	{
	  leaf_ = ME::regTree( reg_ )->getChannel( ig, ix, iy )  ;
	}
    }
  assert( leaf_!=0 );
  setChannel( leaf_ );
}

void
MusEcal::setChannel( MEChannel* leaf )
{
  if( leaf==0 ) 
    {
      cout << "Wrong selection of channel" << endl;
      return;
    }
  if( leaf->ig()<ME::iLMRegion ) return;
  int lmr_ = leaf->getAncestor( ME::iLMRegion )->id();
  _leaf = leaf;
  if( lmr_!=_lmr ) setLMRegion(lmr_); 
 
  cout << "\nCurrent channel: " << _leaf->oneLine() << endl;
}

void
MusEcal::oneLevelUp()
{
  MEChannel* leaf_ = _leaf->m();
  if( leaf_->ig()<ME::iLMRegion )
    {
      cout << "Already at Laser Monitoring Region level" << endl;
      return;
    }
  setChannel( leaf_ );
}

void
MusEcal::refresh()
{
}

bool
MusEcal::isBarrel()
{
  int reg_= ME::ecalRegion( _lmr );
  return ( reg_==ME::iEBM || reg_==ME::iEBP );
}

void
MusEcal::bookHistograms()
{
  if( _histoBooked ) return;

  if( !_ebHistoBooked )
    {
      cout << "Book histograms for barrel" << endl;
      bookEBAPDHistograms();
      bookEBPNHistograms();
      _ebHistoBooked = true;
    }
  if( !_eeHistoBooked )
    {
      cout << "Book histograms for endcaps" << endl;
      bookEEAPDHistograms();
      bookEEPNHistograms(); // JM
      _eeHistoBooked = true;
    }

  _histoBooked = _ebHistoBooked && _eeHistoBooked;

  if( _seq_t==0 )
    {
      _seq_t = new TTree( "Sequences", "Sequences" );
      _seq_t->Branch( "run", &_seq_run, "run/I" );
      _seq_t->Branch( "lb", &_seq_lb, "lb/I" );    
      _seq_t->Branch( "tbeg", &_seq_tbeg, "tbeg/I" );
      _seq_t->Branch( "tlmr", &_seq_tlmr, "tlmr[92]/I" );
      
      _seq_t->SetBranchAddress( "run", &_seq_run );
      _seq_t->SetBranchAddress( "lb", &_seq_lb  );    
      _seq_t->SetBranchAddress( "tbeg", &_seq_tbeg );
      _seq_t->SetBranchAddress( "tlmr", &_seq_tlmr );
    }
}

void
MusEcal::histConfig()
{
  //
  // if there is a config file, modify the specified histogram limits
  //
  
  TString fileroot = TString( getenv("MECONFIG") );
  fileroot += "/";
  fileroot += TString( getenv("MEPERIOD") );
  TString ext_[2] = {"EB","EE"};

  for( int ii=0; ii<2; ii++ )
    {
      TString filename = fileroot;
      filename += "_";
      filename += ext_[ii];
      filename += ".config";

      FILE *test;
      test = fopen(filename,"r");
      char c;
      if( test )
	{
	  ifstream fin(filename);
	  fclose( test );
	  while( (c=fin.peek()) != EOF )
	    {
	      TString ss;
	      fin >> ss;
	      int nbin;
	      fin >> nbin;
	      float min;
	      fin >> min;
	      float max;
	      fin >> max;
	  
	      if( ii==0 )
		{
		  _eb_nbin[ss] = nbin;
		  _eb_min[ss]  = min;
		  _eb_max[ss]  = max;
		}
	      else
		{
		  _ee_nbin[ss] = nbin;
		  _ee_min[ss]  = min;
		  _ee_max[ss]  = max;
		}
	    }
	}
      else
	cout << "WARNING -- Histogram Configuration File " << filename 
	     << " NOT FOUND " 
	     << endl;
	//	abort();
    }
}

int
MusEcal::hist_nbin( TString& str )
{
  if( isBarrel() )
    {
      if( _eb_nbin.count(str)!=0 )
	return _eb_nbin[str];
    }
  else
    {
      if( _ee_nbin.count(str)!=0 )
	return _ee_nbin[str];
    }
  return 0;
}

float
MusEcal::hist_min( TString& str )
{
  assert( hist_nbin(str)!=0 );
  if( isBarrel() )
    return _eb_min[str];
  else
    return _ee_min[str];
}

float
MusEcal::hist_max( TString& str )
{
  assert( hist_nbin(str)!=0 );
  if( isBarrel() )
    return _eb_max[str];
  else
    return _ee_max[str];
}

void
MusEcal::bookEBAPDHistograms()
{
  _febgeom = TFile::Open( ME::path()+"geom/ebgeom.root" );
  assert( _febgeom!=0 );
  _eb_h     = (TH2*) _febgeom->Get("eb");
  _eb_h->SetStats( kFALSE );
  _eb_h->GetXaxis()->SetTitle("ieta");
  _eb_h->GetXaxis()->CenterTitle();
  _eb_h->GetYaxis()->SetTitle("iphi");
  _eb_h->GetYaxis()->CenterTitle();

  _eb_loc_h = (TH2*) _febgeom->Get("eb_loc");
  _eb_loc_h->SetStats( kFALSE );
  _eb_loc_h->GetXaxis()->SetTitle("ix");
  _eb_loc_h->GetXaxis()->CenterTitle();
  _eb_loc_h->GetYaxis()->SetTitle("iy");	      
  _eb_loc_h->GetYaxis()->CenterTitle();

  TH2* h2_;
  TH1* h1_;

  int type = _type;
  //  for( int type=ME::iLaser; type<=ME::iTestPulse; type++ )
    {
      unsigned size_(0);
      TString str0_;
      if( type==ME::iLaser )
	{
	  size_=ME::iSizeAPD;
	  str0_="APD-";
	}
      else if( type==ME::iTestPulse )
	{
	  size_=ME::iSizeTPAPD;
	  str0_="TPAPD-";
	}
      for( unsigned int ii=0; ii<size_; ii++ )
	{
	  TString varName_;
	  if( type==ME::iLaser )          varName_=ME::APDPrimVar[ii];
	  else if( type==ME::iTestPulse ) varName_=ME::TPAPDPrimVar[ii];
	  TString str_=str0_+varName_;
	  h2_ = (TH2*)_eb_h->Clone(str_);	  
	  _eb_m[str_] = h2_;
	  if( _eb_nbin.count(str_)!=0 ) 
	    {
	      h2_->SetMinimum( _eb_min[str_] );
	      h2_->SetMaximum( _eb_max[str_] );
	    }
	  h2_ = (TH2*)_eb_loc_h->Clone(str_+"_loc");
	  _eb_loc_m[str_] = h2_;
	  if( _eb_nbin.count(str_)!=0 ) 
	    {
	      h2_->SetMinimum( _eb_min[str_] );
	      h2_->SetMaximum( _eb_max[str_] );
	    }

	  TString ext_="_1D";
	  h1_ = new TH1F( str_+ext_, varName_, 
			  2448, -0.5, 2447.5 );
	  MECanvasHolder::setHistoStyle( h1_ );
	  h1_->SetStats( kFALSE );
	  h1_->GetXaxis()->SetTitle("SC number (LM numbering)");
	  h1_->GetYaxis()->SetTitle(ME::APDPrimVar[ii]);
	  if( _eb_nbin.count(str_)!=0 ) 
	    {
	      h1_->SetMinimum( _eb_min[str_] );
	      h1_->SetMaximum( 1.1*_eb_max[str_] );
	    }
	  _eb_m[str_+ext_] = h1_;

	  ext_="_VS_CHANNEL";
	  h1_ = new TH1F( str_+ext_, varName_, 900, -0.5, 899.5 );  
	  MECanvasHolder::setHistoStyle( h1_ );
	  h1_->SetStats( kFALSE );
	  h1_->GetXaxis()->SetTitle("Channel number (LM numbering)");
	  h1_->GetYaxis()->SetTitle(ME::APDPrimVar[ii]);
	  if( _eb_nbin.count(str_)!=0 ) 
	    {
	      h1_->SetMinimum( _eb_min[str_] );
	      h1_->SetMaximum( 1.1*_eb_max[str_] );
	    }
	  _eb_loc_m[str_+ext_] = h1_;
	  ext_+="_sel";
	  h1_ = (TH1*) h1_->Clone(str_+ext_);
	  h1_->SetLineColor( kRed );
	  h1_->SetFillColor( 46 );
	  _eb_loc_m[str_+ext_] = h1_;

	  if( _eb_nbin.count(str_)!=0 ) 
	    {
	      ext_="_HIST";
	      h1_ = new TH1F( str_+ext_, varName_, 
			      _eb_nbin[str_], _eb_min[str_], _eb_max[str_] );
	      MECanvasHolder::setHistoStyle( h1_ );
	      _eb_loc_m[str_+ext_] = h1_;
	      ext_+="_sel";
	      h1_ = (TH1*) h1_->Clone(str_+ext_);
	      h1_->SetLineColor( kRed );
	      h1_->SetFillColor( 46 );
	      _eb_loc_m[str_+ext_] = h1_;
	    }
	}
    }
}

void
MusEcal::bookEEAPDHistograms()
{
  _feegeom = TFile::Open( ME::path()+"geom/eegeom.root" );
  assert( _feegeom!=0 );
  _ee_h     = (TH2*) _feegeom->Get("ee");
  _ee_h->SetStats( kFALSE );
  _ee_h->GetXaxis()->SetTitle("ix");
  _ee_h->GetXaxis()->CenterTitle();
  _ee_h->GetYaxis()->SetTitle("iy");
  _ee_h->GetYaxis()->CenterTitle();

  _ee_loc_h[0] = 0;
  for( int isect=1; isect<=9; isect++ )
    {
      TString sect_("ee_S");
      sect_+=isect;
      _ee_loc_h[isect] = (TH2*) _feegeom->Get(sect_);
      _ee_loc_h[isect]->SetStats( kFALSE );
      _ee_loc_h[isect]->GetXaxis()->SetTitle("ix");
      _ee_loc_h[isect]->GetXaxis()->CenterTitle();
      _ee_loc_h[isect]->GetYaxis()->SetTitle("iy");	      
      _ee_loc_h[isect]->GetYaxis()->CenterTitle();
    }

  // get the master tree for EE-plus (sectors 1 to 9)
  MEChannel* tree = ME::regTree( ME::iEEP );

  TH2* h2_;
  TH1* h1_;

  int type = _type;
  //  for( int type=ME::iLaser; type<=ME::iTestPulse; type++ )
  unsigned size_(0);
  TString str0_;
  if( type==ME::iLaser )
    {
      size_=ME::iSizeAPD;
      str0_="APD-";
    }
  else if( type==ME::iTestPulse )
    {
      size_=ME::iSizeTPAPD;
      str0_="TPAPD-";
    }
  for( unsigned int ii=0; ii<size_; ii++ )
    {
      TString varName_;
      if( type==ME::iLaser )          varName_=ME::APDPrimVar[ii];
      else if( type==ME::iTestPulse ) varName_=ME::TPAPDPrimVar[ii];
      TString str_=str0_+varName_;
      h2_ = (TH2*)_ee_h->Clone(str_);	  
      _ee_m[str_] = h2_;
      if( _ee_nbin.count(str_)!=0 ) 
	{
	  h2_->SetMinimum( _ee_min[str_] );
	  h2_->SetMaximum( _ee_max[str_] );
	}
      for( int isect=1; isect<=9; isect++ )
	{
	  TString str__ = str_;
	  str__+="_";
	  str__+=isect;
	  h2_ = (TH2*)_ee_loc_h[isect]->Clone(str__+"_loc");
	  _ee_loc_m[str__] = h2_;
	  if( _ee_nbin.count(str_)!=0 ) 
	    {
	      h2_->SetMinimum( _ee_min[str_] );
	      h2_->SetMaximum( _ee_max[str_] );
	    }

	  TString ext_="_VS_CHANNEL";
	  
	  // trick to get the correct number of bins...
	  MEChannel* tree_ = tree->getDescendant( ME::iSector, isect );
	  vector< MEChannel* > vec_;
	  tree_->getListOfDescendants( ME::iCrystal, vec_ );
	  int nbin_ = vec_.size();

	  //cout << "test EE2 vec size="<<nbin_ << endl;
	  float xmin_ = -0.5; float xmax_ = nbin_-0.5;

	  h1_ = new TH1F( str__+ext_, varName_, nbin_, xmin_, xmax_ );  
	  MECanvasHolder::setHistoStyle( h1_ );
	  h1_->SetStats( kFALSE );
	  h1_->GetXaxis()->SetTitle("Channel number (LM numbering)");
	  h1_->GetYaxis()->SetTitle(ME::APDPrimVar[ii]);
	  if( _ee_nbin.count(str_)!=0 ) 
	    {
	      h1_->SetMinimum( _ee_min[str_] );
	      h1_->SetMaximum( 1.1*_ee_max[str_] );
	    }
	  _ee_loc_m[str__+ext_] = h1_;
	  ext_+="_sel";
	  h1_ = (TH1*) h1_->Clone(str__+ext_);
	  h1_->SetLineColor( kRed );
	  h1_->SetFillColor( 46 );
	  _ee_loc_m[str__+ext_] = h1_;

	  if( _ee_nbin.count(str_)!=0 ) 
	    {
	      ext_="_HIST";
	      h1_ = new TH1F( str__+ext_, varName_, 
			      _ee_nbin[str_], _ee_min[str_], _ee_max[str_] );
	      MECanvasHolder::setHistoStyle( h1_ );
	      _ee_loc_m[str__+ext_] = h1_;
	      ext_+="_sel";
	      h1_ = (TH1*) h1_->Clone(str__+ext_);
	      h1_->SetLineColor( kRed );
	      h1_->SetFillColor( 46 );
	      _ee_loc_m[str__+ext_] = h1_;
	    }
	}
	  
      TString ext_="_1D";
      h1_ = new TH1F( str_+ext_, varName_, 
		      624, -0.5, 623.5 );
      MECanvasHolder::setHistoStyle( h1_ );
      h1_->SetStats( kFALSE );
      h1_->GetXaxis()->SetTitle("SC number (LM numbering)");
      h1_->GetYaxis()->SetTitle(ME::APDPrimVar[ii]);
      if( _ee_nbin.count(str_)!=0 ) 
	{
	  h1_->SetMinimum( _ee_min[str_] );
	  h1_->SetMaximum( 1.1*_ee_max[str_] );
	}
      _ee_m[str_+ext_] = h1_;

    }
}

void
MusEcal::bookEBPNHistograms()
{
  TH1* h1_;

  int type = _type;
  //  for( int type=ME::iLaser; type<=ME::iTestPulse; type++ )
    {
      unsigned size_(0);
      TString str0_;
      if( type==ME::iLaser )
	{
	  size_=ME::iSizePN;
	  str0_="PN-";
	}
      else if( type==ME::iTestPulse )
	{
	  size_=ME::iSizeTPPN;
	  str0_="TPPN-";
	}
      for( unsigned int ii=0; ii<size_; ii++ )
	{
	  TString varName_, str_;
	  if( type==ME::iLaser ) varName_=ME::PNPrimVar[ii];
	  else if( type==ME::iTestPulse ) varName_=ME::TPPNPrimVar[ii];
	  str_=str0_+varName_;
	  
	  // global histogram
	  h1_ = new TH1F( str_, varName_, 648, -0.5, 647.5 );
	  MECanvasHolder::setHistoStyle( h1_ );
	  h1_->SetStats( kFALSE );
	  h1_->GetXaxis()->SetTitle("PN number (LM numbering)");
	  h1_->GetYaxis()->SetTitle( varName_ );
	  if( _ee_nbin.count(str_)!=0 ) 
	    {
	      h1_->SetMinimum( _eb_min[str_] );
	      h1_->SetMaximum( 1.1*_eb_max[str_] );
	    }
	  _eb_m[str_] = h1_;
	  
	  // local histogram
	  TString ext_ = "_LOCAL";
	  h1_ = new TH1F( str_+ext_, varName_, 
			  18, -0.5, 17.5 ); 
	  MECanvasHolder::setHistoStyle( h1_ );
	  h1_->SetStats( kFALSE );
	  h1_->GetXaxis()->SetTitle("PN number (LM numbering)");
	  h1_->GetYaxis()->SetTitle( varName_ );
	  if( _ee_nbin.count(str_)!=0 ) 
	    {
	      h1_->SetMinimum( _eb_min[str_] );
	      h1_->SetMaximum( 1.1*_eb_max[str_] );
	    }
	  _eb_loc_m[str_+ext_] = h1_;
	}
    }
}
void
MusEcal::bookEEPNHistograms()  // JM
{

  TH1* h1_;
  
  int type = _type;
  //  for( int type=ME::iLaser; type<=ME::iTestPulse; type++ )
  {
    unsigned size_(0);
    TString str0_;
    if( type==ME::iLaser )
      {
	size_=ME::iSizePN;
	str0_="PN-";
      }
    else if( type==ME::iTestPulse )
      {
	size_=ME::iSizeTPPN;
	str0_="TPPN-";
      }
    for( unsigned int ii=0; ii<size_; ii++ )
      {
	TString varName_, str_;
	if( type==ME::iLaser ) varName_=ME::PNPrimVar[ii];
	else if( type==ME::iTestPulse ) varName_=ME::TPPNPrimVar[ii];
	str_=str0_+varName_;
	
	// global histogram
	h1_ = new TH1F( str_, varName_, 152, -0.5, 151.5 );
	MECanvasHolder::setHistoStyle( h1_ );
	h1_->SetStats( kFALSE );
	h1_->GetXaxis()->SetTitle("PN number (LM numbering)");
	h1_->GetYaxis()->SetTitle( varName_ );
	if( _ee_nbin.count(str_)!=0 ) 
	  {
	    h1_->SetMinimum( _ee_min[str_] );
	    h1_->SetMaximum( 1.1*_ee_max[str_] );
	  }
	_ee_m[str_] = h1_;
	
	// local histogram
	TString ext_ = "_LOCAL";
	int thislmr;
	if(_lmr>72 && _lmr<=92) thislmr=_lmr;
	else thislmr=73;

	std::vector<int> vecmod=MEEEGeom::lmmodFromLmr( thislmr );
	int h1_nbins=vecmod.size()*2;
	double h1_min=-0.5;
	double h1_max=-0.5+double(h1_nbins)*1.0;
	
	h1_ = new TH1F( str_+ext_, varName_, 
			h1_nbins, h1_min, h1_max ); 
	
	MECanvasHolder::setHistoStyle( h1_ );
	h1_->SetStats( kFALSE );
	h1_->GetXaxis()->SetTitle("PN number (LM numbering)");
	h1_->GetYaxis()->SetTitle( varName_ );
	if( _ee_nbin.count(str_)!=0 ) 
	  {
	    h1_->SetMinimum( _ee_min[str_] );
	    h1_->SetMaximum( 1.1*_ee_max[str_] );
	  }
	_ee_loc_m[str_+ext_] = h1_;
      }
  }
}

void
MusEcal::fillHistograms()
{
  if( !_histoBooked ) bookHistograms();
  fillEBGlobalHistograms();
  fillEEGlobalHistograms();
  if( isBarrel() )
    {
      fillEBLocalHistograms();
    }
  else
    {
      fillEELocalHistograms();
    }
}

void
MusEcal::fillEBGlobalHistograms()
{
  if( !_histoBooked ) bookHistograms();
  cout << "Filling EB Global Histograms";
  // cleaning
  for( map<TString,TH1*>::iterator it=_eb_m.begin();
       it!=_eb_m.end(); ++it )
    {
      it->second->Reset();
    } 

  // filling the 2D histogram

  TString rundir_;
  MERunManager* firstMgr_ = _runMgr.begin()->second;
  if( firstMgr_!=0 )
    {
      MERun* firstRun_ = firstMgr_->curRun();
      if( firstRun_!=0 ) 
	{
	  rundir_ += firstRun_->rundir();
	}
    }
  TString titleW;

  titleW = ME::type[_type];
  if( _type==ME::iLaser )
    {
      titleW+=" "; titleW+=ME::color[_color];
    }
  titleW+=" ECAL Barrel";
  titleW+=" APD XXXXX";
  titleW+=" "; titleW+=rundir_;

  unsigned size_(0);
  TString str0_;
  int table_(0);

  vector< MEChannel* > vec;
  for( int ism=1; ism<=36; ism++ )
    {
      cout << "." << flush;
      int idcc=MEEBGeom::dccFromSm( ism );
      int ilmr;
      MERunManager* mgr_;
      MERun* run_;
      for( int side=0; side<2; side++ )
	{
	  ilmr=ME::lmr( idcc, side );
	  mgr_=runMgr( ilmr );
	  if( mgr_==0 ) continue;
	  run_= mgr_->curRun();
	  if( run_==0 ) continue;

	  // first APD
	  if( _type==ME::iLaser )
	    {
	      size_=ME::iSizeAPD;
	      str0_="APD-";
	      table_=ME::iLmfLaserPrim;
	    }
	  else if( _type==ME::iTestPulse )
	    {
	      size_=ME::iSizeTPAPD;
	      str0_="TPAPD-";
	      table_=ME::iLmfTestPulsePrim;
	    }

	  vec.clear();
	  mgr_->tree()->getListOfChannels( vec );
	  for( unsigned int jj=0; jj<size_; jj++ )
	    {
	      TString varName_;
	      if( _type==ME::iLaser )
		{
		  varName_=ME::APDPrimVar[jj];
		}
	      else if( _type==ME::iTestPulse )
		{
		  varName_=ME::TPAPDPrimVar[jj];
		}
	      for( unsigned int ii=0; ii<vec.size(); ii++ )
		{
		  MEChannel* leaf_ = vec[ii];
		  int ieta = leaf_->ix();
		  int iphi = leaf_->iy();
		  MEEBGeom::XYCoord ixy = MEEBGeom::localCoord( ieta, iphi );
		  int ix = ixy.first;
		  int iy = ixy.second;
		  float val = run_->getVal( table_, jj, ix, iy );
		  TString str_=str0_+varName_;
		  TH2* h2_;
		  h2_ = (TH2*) _eb_m[str_];
		  TString title_ = titleW;
		  title_.ReplaceAll("XXXXX",str_);
		  h2_->SetTitle(title_);
		  h2_->Fill( ieta, iphi, val ); 
		  TH1* h1_;
		  TString ext_="_1D";
		  h1_ = (TH1*) _eb_m[str_+ext_];
		  h1_->SetTitle(title_);
		  int ival = ii + ((ilmr-1)/2)*1700;
		  if( side==1 ) ival += 900;
		  h1_->Fill( ival/25, val/25 );		  
		}
	    }

	  // then PNs
	  if( _type==ME::iLaser )
	    {
	      size_=ME::iSizePN;
	      str0_="PN-";
	      table_=ME::iLmfLaserPnPrim;
	    }
	  else if( _type==ME::iTestPulse )
	    {
	      size_=ME::iSizeTPPN;
	      str0_="TPPN-";
	      table_=ME::iLmfTestPulsePnPrim;
	    }
	  vec.clear();
	  mgr_->tree()->getListOfDescendants( ME::iLMModule, vec );
	  for( unsigned int jj=0; jj<size_; jj++ )
	    {
	      TString varName_;
	      if( _type==ME::iLaser )
		{
		  varName_=ME::PNPrimVar[jj];
		}
	      else if( _type==ME::iTestPulse )
		{
		  varName_=ME::TPPNPrimVar[jj];
		}
	      for( unsigned int ii=0; ii<vec.size(); ii++ )
		{
		  MEChannel* leaf_ = vec[ii];
		  int ilm = leaf_->id();
		  TString str_=str0_+varName_;
		  TH1* h1_;
		  h1_ = (TH1*) _eb_m[str_];
		  TString title_ = titleW;
		  title_.ReplaceAll("APD","PN");
		  title_.ReplaceAll("XXXXX",str_);
		  h1_->SetTitle(title_);
		  for( int ipn=0; ipn<2; ipn++ )
		    {
		      float val = run_->getVal( table_, jj, ilm, ipn );
		      int ival = 2*((ilm-1)/2) + ipn + ((ilmr-1)/2)*18;
		      if( ilm%2==0 ) ival+=10;
		      h1_->Fill( ival, val );
		    }		  
		}
	    }
	}
    }
  cout << " Done." << endl;
}

void
MusEcal::fillEBLocalHistograms()
{
  if( !_histoBooked ) bookHistograms();
  cout << "Filling EB Local Histograms";
  for( map<TString,TH1*>::iterator it=_eb_loc_m.begin();
       it!=_eb_loc_m.end(); ++it )
    {
      it->second->Reset();
    } 
  MERunManager* mgr = curMgr();
  if( mgr==0 ) return;
  MERun* run_ = mgr->curRun();
  TString rundir="No Data";
  if( run_!=0 ) rundir = run_->rundir();

  TString titleW;
  titleW = ME::type[_type];
  if( _type==ME::iLaser )
    {
      titleW+=" "; titleW+=ME::color[_color];
    }
  titleW+=" "; titleW+="YYYYY";
  titleW+=" APD XXXXX";
  titleW+=" "; titleW+=rundir;
  TString title_;

  MEChannel* l_[2] = {0,0};
  l_[0] = mgr->tree();
  l_[1] = _leaf;

  unsigned size_(0);
  TString str0_;
  int table_(0);

  if( _type==ME::iLaser )
    {
      size_=ME::iSizeAPD;
      str0_="APD-";
      table_=ME::iLmfLaserPrim;
    }
  else if( _type==ME::iTestPulse )
    {
      size_=ME::iSizeTPAPD;
      str0_="TPAPD-";
      table_=ME::iLmfTestPulsePrim;
    }

  for( unsigned int jj=0; jj<size_; jj++ )
    {
      TString varName_;
      if( _type==ME::iLaser )
	{
	  varName_=ME::APDPrimVar[jj];
	}
      else if( _type==ME::iTestPulse )
	{
	  varName_=ME::TPAPDPrimVar[jj];
	}
      TH2* h2_(0);
      TH1* h1_[2]={0,0};
      TH1* h1_chan[2]={0,0};
      TString str_=str0_+varName_;
      h2_ = (TH2*) _eb_loc_m[str_];
      title_= titleW;
      title_.ReplaceAll("XXXXX",varName_);
      //  title_.ReplaceAll("YYYYY",_leaf->oneWord( ME::iLMRegion ) );
      title_.ReplaceAll("YYYYY",_leaf->oneWord() );
      h2_->SetTitle( title_ );

      title_= titleW;
      title_.ReplaceAll("XXXXX",varName_);
      title_.ReplaceAll("YYYYY",_leaf->oneWord() );
      TString ext_="_HIST";
      if( _eb_loc_m.count(str_+ext_)!=0 )
	{		  
	  h1_[0] = (TH1*) _eb_loc_m[str_+ext_];
	  h1_[0]->SetTitle( title_ );
	  ext_+="_sel";
	  h1_[1] = (TH1*) _eb_loc_m[str_+ext_];
	  h1_[1]->SetTitle( title_ );
	}

      ext_="_VS_CHANNEL";
      h1_chan[0] = (TH1*) _eb_loc_m[str_+ext_];
      h1_chan[0]->SetTitle( title_ );
      ext_+="_sel";
      h1_chan[1] = (TH1*) _eb_loc_m[str_+ext_];
      h1_chan[1]->SetTitle( title_ );

      if( run_!=0 ) 
	{
	  vector< MEChannel* > vec[2];
	  l_[0]->getListOfChannels( vec[0] );
	  l_[1]->getListOfChannels( vec[1] );
	  for( unsigned int ii=0; ii<vec[0].size(); ii++ )
	    {
	      MEChannel* leaf_ = vec[0][ii];
	      int ieta = leaf_->ix();
	      int iphi = leaf_->iy();
	      MEEBGeom::XYCoord ixy = MEEBGeom::localCoord( ieta, iphi );
	      int ix = ixy.first;
	      int iy = ixy.second;
	      float val = run_->getVal( table_, jj, ix, iy );
	      if( h1_chan[0]!=0 ) h1_chan[0]->Fill( ii, val );
	      if( h1_[0]!=0 )     h1_[0]->Fill( val );
	      vector<MEChannel*>::iterator it_ = 
		find( vec[1].begin(), vec[1].end(), leaf_ );
	      if( it_!=vec[1].end() )
		{
		  if( h1_chan[1]!=0 ) h1_chan[1]->Fill( ii, val );
		  if( h1_[1]!=0 )     h1_[1]->Fill( val );
		  // test
		  if(_leaf->ig()>ME::iLMRegion )
		    {
		      TAxis* xaxis_ = h2_->GetXaxis();		  
		      TAxis* yaxis_ = h2_->GetYaxis();		  
		      int a1_ = xaxis_->GetFirst();
		      int a2_ = xaxis_->GetLast();
		      int b1_ = yaxis_->GetFirst();
		      int b2_ = yaxis_->GetLast();
		      float max_ = h2_->GetMaximum();
		      for( int ix_=-1; ix_>=xaxis_->GetBinCenter(a1_); ix_-- )
			{
			  h2_->Fill( ix_, iy, max_ );
			}
		      for( int ix_=85; ix_<=xaxis_->GetBinCenter(a2_); ix_++ )
			{
			  h2_->Fill( ix_, iy, max_ );
			}
		      for( int iy_=-1; iy_>=yaxis_->GetBinCenter(b1_); iy_-- )
			{
			  h2_->Fill( ix, iy_, max_ );
			}
		      for( int iy_=20; iy_<=yaxis_->GetBinCenter(b2_); iy_++ )
			{
			  h2_->Fill( ix, iy_, max_ );
			}
		    }
		}

	      h2_->Fill( ix, iy, val );
	    }
	}
    }
  // Now PN
  if( _type==ME::iLaser )
    {
      size_=ME::iSizePN;
      str0_="PN-";
      table_=ME::iLmfLaserPnPrim;
    }
  else if( _type==ME::iTestPulse )
    {
      size_=ME::iSizeTPPN;
      str0_="TPPN-";
      table_=ME::iLmfTestPulsePnPrim;
    }

  for( unsigned int jj=0; jj<size_; jj++ )
    {
      TString varName_;
      if( _type==ME::iLaser )
	{
	  varName_=ME::PNPrimVar[jj];
	}
      else if( _type==ME::iTestPulse )
	{
	  varName_=ME::TPPNPrimVar[jj];
	}
      TH1* h1_;
      TString str_ = str0_+varName_;
      TString ext_ = "_LOCAL";
      h1_ = (TH1*) _eb_loc_m[str_+ext_];
      title_= titleW;
      title_.ReplaceAll("APD","PN");
      title_.ReplaceAll("XXXXX",varName_);
      title_.ReplaceAll("YYYYY",_leaf->oneWord( ME::iLMRegion ) );
      h1_->SetTitle( title_ );

      if( run_!=0 )
	{
	  vector< MEChannel* > vec;
	  mgr->tree()->getListOfDescendants( ME::iLMModule, vec );
	  for( unsigned int ii=0; ii<vec.size(); ii++ )
	    {
	      MEChannel* leaf_ = vec[ii];
	      int ilm = leaf_->id();
	      for( int ipn=0; ipn<2; ipn++ )
		{
		  float val = run_->getVal( table_, jj, ilm, ipn );
		  int ival = 2*((ilm-1)/2) + ipn ;
		  if( ilm%2==0 ) ival+=10;
		  h1_->Fill( ival, val );
		}
	    }
	}
    }
  cout << "...Done." << endl;
}

void
MusEcal::fillEEGlobalHistograms()
{  
  if( !_histoBooked ) bookHistograms();
  cout << "Filling EE Global Histograms";
  // cleaning
  for( map<TString,TH1*>::iterator it=_ee_m.begin();
       it!=_ee_m.end(); ++it )
    {
      it->second->Reset();
    } 

  // filling the 2D histogram
  TString rundir_;
  MERunManager* firstMgr_ = _runMgr.begin()->second;
  if( firstMgr_!=0 )
    {
      MERun* firstRun_ = firstMgr_->curRun();
      if( firstRun_!=0 ) 
	{
	  rundir_ += firstRun_->rundir();
	}
    }
  TString titleW;

  titleW = ME::type[_type];
  if( _type==ME::iLaser )
    {
      titleW+=" "; titleW+=ME::color[_color];
    }
  titleW+=" ECAL EndCap";
  titleW+=" APD XXXXX";
  titleW+=" "; titleW+=rundir_;

  unsigned size_(0);
  TString str0_;
  int table_(0);

  vector< MEChannel* > vec;
  int ilmr, ireg, idcc, isect, side;
  int iterchan=0;
  for( ilmr=73; ilmr<=92; ilmr++ )
    {
      cout << "." << flush;
      int iz = 1;
      if( ilmr>82 ) iz = -1;
      ME::regionAndSector( ilmr, ireg, isect, idcc, side );
      MERunManager* mgr_;
      MERun* run_;
      mgr_=runMgr( ilmr );
      if( mgr_==0 ) continue;
      //    int ism = MEEEGeom::smFromDcc( idcc );
      run_= mgr_->curRun();
      if( run_==0 ) continue;

      // first APD
      if( _type==ME::iLaser )
	{
	  size_=ME::iSizeAPD;
	  str0_="APD-";
	  table_=ME::iLmfLaserPrim;
	}
      else if( _type==ME::iTestPulse )
	{
	  size_=ME::iSizeTPAPD;
	  str0_="TPAPD-";
	  table_=ME::iLmfTestPulsePrim;
	}
      
      vec.clear();
      mgr_->tree()->getListOfChannels( vec );

      for( unsigned int jj=0; jj<size_; jj++ )
	{
	  TString varName_;
	  if( _type==ME::iLaser )
	    {
	      varName_=ME::APDPrimVar[jj];
	    }
	  else if( _type==ME::iTestPulse )
	    {
	      varName_=ME::TPAPDPrimVar[jj];
	    }
	  TString str_=str0_+varName_;
	  TH2* h2_;
	  h2_ = (TH2*) _ee_m[str_];
	  if( h2_==0 ) 
	    {
	      cout << "non existing histogram " << str_ << endl;
	      continue;
	    }
	  TString title_ = titleW;
	  title_.ReplaceAll("XXXXX",str_);
	  h2_->SetTitle(title_);
	  TH1* h1_;
	  TString ext_="_1D";	      
	  h1_ = (TH1*) _ee_m[str_+ext_];
	  if( h1_==0 ) 
	    {
	      cout << "non existing histogram " << str_ << endl;
	      continue;	  
	    }
	  h1_->SetTitle(title_);

	  for( unsigned int ii=0; ii<vec.size(); ii++ )
	    {
	      MEChannel* leaf_ = vec[ii];
	      int ix = leaf_->ix();
	      int iy = leaf_->iy();
	      float val = run_->getVal( table_, jj, ix, iy );	      
	      h2_->Fill( ix, iz*iy, val ); 

	      //int isc = MEEEGeom::sc( iX, iY );
	      //int test= MEEEGeom::crystal_in_sc( ix, iy );
	      //cout << "EEchan iterchan:"<<iterchan<<" "<<isc<<" "<<test<< endl;
	      float ival=ii+iterchan;
	      if(ii==vec.size()-1) iterchan+=vec.size();
	      
	      h1_->Fill( ival/25, val/25 ); // JM ????
	    }
	}

      // then PNs //JM 

      if( _type==ME::iLaser )
	{
	  size_=ME::iSizePN;
	  str0_="PN-";
	  table_=ME::iLmfLaserPnPrim;
	}
      else if( _type==ME::iTestPulse )
	{
	  size_=ME::iSizeTPPN;
	  str0_="TPPN-";
	  table_=ME::iLmfTestPulsePnPrim;
	}
      vec.clear();
      mgr_->tree()->getListOfDescendants( ME::iLMModule, vec );

      for( unsigned int jj=0; jj<size_; jj++ )
	{
	  TString varName_;
	  if( _type==ME::iLaser )
	    {
	      varName_=ME::PNPrimVar[jj];
	    }
	  else if( _type==ME::iTestPulse )
	    {
	      varName_=ME::TPPNPrimVar[jj];
	    }

	  for( unsigned int ii=0; ii<vec.size(); ii++ )
	    {
	      MEChannel* leaf_ = vec[ii];
	      int ilm = leaf_->id();
	      
	      TString str_=str0_+varName_;
	      TH1* h1_;
	      h1_ = (TH1*) _ee_m[str_];
	      TString title_ = titleW;
	      title_.ReplaceAll("APD","PN");
	      title_.ReplaceAll("XXXXX",str_);
	      h1_->SetTitle(title_);
	      for( int ipn=0; ipn<2; ipn++ )
		{
		  float val = run_->getVal( table_, jj, ilm, ipn );
		  int ival0=0;
		  
		  for(int iit=73;iit<ilmr;iit++){
		    std::vector<int> vecmod=MEEEGeom::lmmodFromLmr( iit );
		    int ntoadd=vecmod.size()*2;
		    ival0+=ntoadd;
		  }
		  
		  std::vector<int> thisvecmod=MEEEGeom::lmmodFromLmr( ilmr );
		  int ilm0=thisvecmod[0];
		  
		  int ival = ival0 + 2*(ilmr-ilm0) + ipn ;
		  
		  h1_->Fill( ival, val );
		}
	    }
	}
    }
  cout << " Done." << endl;
}

void
MusEcal::fillEELocalHistograms()
{
  if( !_histoBooked ) bookHistograms();
  if( isBarrel() ) return;
  //  int reg_= ME::ecalRegion( _lmr );
  //  int iz=1;
  //  if( reg_==ME::iEEM ) iz=-1;
  
  cout << "Filling EE Local Histograms";
  for( map<TString,TH1*>::iterator it=_ee_loc_m.begin();
       it!=_ee_loc_m.end(); ++it )
    {
      it->second->Reset();
    } 
  MERunManager* mgr = curMgr();
  if( mgr==0 ) return;
  MERun* run_ = mgr->curRun();
  TString rundir="No Data";
  if( run_!=0 ) rundir = run_->rundir();

  TString titleW;
  titleW = ME::type[_type];
  if( _type==ME::iLaser )
    {
      titleW+=" "; titleW+=ME::color[_color];
    }
  titleW+=" "; titleW+="YYYYY";
  titleW+=" APD XXXXX";
  titleW+=" "; titleW+=rundir;
  TString title_;

  cout << "GHM DBG -- " << titleW << endl;

  unsigned size_(0);
  TString str0_;
  int table_(0);

  if( _type==ME::iLaser )
    {
      size_=ME::iSizeAPD;
      str0_="APD-";
      table_=ME::iLmfLaserPrim;
    }
  else if( _type==ME::iTestPulse )
    {
      size_=ME::iSizeTPAPD;
      str0_="TPAPD-";
      table_=ME::iLmfTestPulsePrim;
    }

  cout << "GHM DBG -- " << _leaf->oneLine() << endl;

  MEChannel* l_[2] = {0,0};
  l_[1] = _leaf;
  //  l_[0] = mgr->tree();
  l_[0] = _leaf->getAncestor( ME::iSector );

  int isect = l_[0]->id();
  if( isect>9 ) isect-=9;
  cout << "GHM DBG isect= " << isect << endl;

  for( unsigned int jj=0; jj<size_; jj++ )
    {
      TString varName_;
      if( _type==ME::iLaser )
	{
	  varName_=ME::APDPrimVar[jj];
	}
      else if( _type==ME::iTestPulse )
	{
	  varName_=ME::TPAPDPrimVar[jj];
	}
      TH2* h2_(0);
      TH1* h1_[2]={0,0};
      TH1* h1_chan[2]={0,0};
      TString str_=str0_;
      str_+=varName_;
      str_+="_";
      str_+=isect;
      h2_ = (TH2*) _ee_loc_m[str_];
      title_= titleW;
      title_.ReplaceAll("XXXXX",varName_);
      //  title_.ReplaceAll("YYYYY",_leaf->oneWord( ME::iLMRegion ) );
      title_.ReplaceAll("YYYYY",_leaf->oneWord() );
      h2_->SetTitle( title_ );

      title_= titleW;
      title_.ReplaceAll("XXXXX",varName_);
      title_.ReplaceAll("YYYYY",_leaf->oneWord() );
      TString ext_="_HIST";
      if( _ee_loc_m.count(str_+ext_)!=0 )
	{		  
	  h1_[0] = (TH1*) _ee_loc_m[str_+ext_];
	  h1_[0]->SetTitle( title_ );
	  ext_+="_sel";
	  h1_[1] = (TH1*) _ee_loc_m[str_+ext_];
	  h1_[1]->SetTitle( title_ );
	}

      ext_="_VS_CHANNEL";
      if( _ee_loc_m.count(str_+ext_)!=0 )
	{		  
	  h1_chan[0] = (TH1*) _ee_loc_m[str_+ext_];
	  h1_chan[0]->SetTitle( title_ );
	  ext_+="_sel";
	  h1_chan[1] = (TH1*) _ee_loc_m[str_+ext_];
	  h1_chan[1]->SetTitle( title_ );
	}

      if( run_!=0 ) 
	{
	  vector< MEChannel* > vec[2];
	  l_[0]->getListOfChannels( vec[0] );
	  l_[1]->getListOfChannels( vec[1] );
	  for( unsigned int ii=0; ii<vec[0].size(); ii++ )
	    {
	      MEChannel* leaf_ = vec[0][ii];
	      //	      int ieta = leaf_->ix();
	      //	      int iphi = leaf_->iy();
	      //	      MEEBGeom::XYCoord ixy = MEEBGeom::localCoord( ieta, iphi );
	      int ix = leaf_->ix();
	      int iy = leaf_->iy();
	      float val = run_->getVal( table_, jj, ix, iy );
	      if( h1_chan[0]!=0 ) h1_chan[0]->Fill( ii, val );
	      if( h1_[0]!=0 )     h1_[0]->Fill( val );
	      vector<MEChannel*>::iterator it_ = 
		find( vec[1].begin(), vec[1].end(), leaf_ );
	      if( it_!=vec[1].end() )
		{
		  if( h1_chan[1]!=0 ) h1_chan[1]->Fill( ii, val );
		  if( h1_[1]!=0 )     h1_[1]->Fill( val );
		  // test
// 		  if(_leaf->ig()>ME::iLMRegion )
// 		    {
// 		      TAxis* xaxis_ = h2_->GetXaxis();		  
// 		      TAxis* yaxis_ = h2_->GetYaxis();		  
// 		      int a1_ = xaxis_->GetFirst();
// 		      int a2_ = xaxis_->GetLast();
// 		      int b1_ = yaxis_->GetFirst();
// 		      int b2_ = yaxis_->GetLast();
// 		      float max_ = h2_->GetMaximum();
// 		      for( int ix_=-1; ix_>=xaxis_->GetBinCenter(a1_); ix_-- )
// 			{
// 			  h2_->Fill( ix_, iy, max_ );
// 			}
// 		      for( int ix_=85; ix_<=xaxis_->GetBinCenter(a2_); ix_++ )
// 			{
// 			  h2_->Fill( ix_, iy, max_ );
// 			}
// 		      for( int iy_=-1; iy_>=yaxis_->GetBinCenter(b1_); iy_-- )
// 			{
// 			  h2_->Fill( ix, iy_, max_ );
// 			}
// 		      for( int iy_=20; iy_<=yaxis_->GetBinCenter(b2_); iy_++ )
// 			{
// 			  h2_->Fill( ix, iy_, max_ );
// 			}
// 		    }
		}


	      //	      cout << "GHM DBG ix/iy " << ix << "/" << iy << "/" << val << endl;
	      h2_->Fill( ix, iy, val );
	    }
	}
    }
  // Now PN
  if( _type==ME::iLaser )
    {
      size_=ME::iSizePN;
      str0_="PN-";
      table_=ME::iLmfLaserPnPrim;
    }
  else if( _type==ME::iTestPulse )
    {
      size_=ME::iSizeTPPN;
      str0_="TPPN-";
      table_=ME::iLmfTestPulsePnPrim;
    }

  for( unsigned int jj=0; jj<size_; jj++ )
    {
      TString varName_;
      if( _type==ME::iLaser )
	{
	  varName_=ME::PNPrimVar[jj];
	}
      else if( _type==ME::iTestPulse )
	{
	  varName_=ME::TPPNPrimVar[jj];
	}
      TH1* h1_;
      TString str_ = str0_+varName_;
      TString ext_ = "_LOCAL";
      h1_ = (TH1*) _eb_loc_m[str_+ext_];
      title_= titleW;
      title_.ReplaceAll("APD","PN");
      title_.ReplaceAll("XXXXX",varName_);
      title_.ReplaceAll("YYYYY",_leaf->oneWord( ME::iLMRegion ) );
      h1_->SetTitle( title_ );

      if( run_!=0 )
	{
	  vector< MEChannel* > vec;
	  mgr->tree()->getListOfDescendants( ME::iLMModule, vec );
	  for( unsigned int ii=0; ii<vec.size(); ii++ )
	    {
	      MEChannel* leaf_ = vec[ii];
	      int ilm = leaf_->id();
	      for( int ipn=0; ipn<2; ipn++ )
		{
		  float val = run_->getVal( table_, jj, ilm, ipn );
		  int ival = 2*((ilm-1)/2) + ipn ;
		  if( ilm%2==0 ) ival+=10;
		  h1_->Fill( ival, val );
		}
	    }
	}
    }

//   for( map<TString,TH1*>::iterator it=_ee_loc_m.begin();
//        it!=_ee_loc_m.end(); ++it )
//     {
//       cout << "ee_loc_m " << it->first << endl;
//     } 
  cout << "...Done." << endl;
}

METimeInterval* 
MusEcal::pnIntervals(  std::vector<int>& iapdopn , MEChannel* leaf )
{

  MEChannel* modLeaf = leaf->getAncestor( ME::iLMModule );

  if( ( _color<0 || _color>=ME::iSizeC ) || _type != ME::iLaser )
    {
      cout << "PN Validity Intervals: Only for LASER data" << endl;
      return 0;
    } 
  
  if( leaf->ig()<ME::iLMRegion )
    {
      cout << "Leaf must be at least with Side granularity" << endl;
      return 0;
    } 

  if( _pnintervals[_color].count( modLeaf ) ){
    iapdopn=_pnvarmap[_color][modLeaf];
    return _pnintervals[_color][modLeaf];
  }

  buildPNIntervals( iapdopn, leaf );
  
  assert( _pnintervals[_color].count( modLeaf )==1 );
  return _pnintervals[_color][modLeaf];
}


MEIntervals* 
MusEcal::intervals( MEChannel *leaf )
{

  if(_debug) cout<< "Entering intervals "<<_type<<" "<<_color<< endl;

  if( leaf->ig()<ME::iLMRegion )
    {
      cout << "Leaf must be at least with Side granularity" << endl;
      return 0;
    } 
  
  if( _color>=0 && _color<ME::iSizeC && _type == ME::iLaser )
    return mtqIntervals( true, leaf );
  else if( _color==0 && _type == ME::iTestPulse )
    return tpIntervals( leaf );
  else{
    cout << "Validity Intervals: Only for LASER or TP data" << endl;
    return 0;
  } 
  
}

MEIntervals* 
MusEcal::mtqIntervals( bool createCor , MEChannel *leaf )
{

  if(_debug) cout<< "Entering mtqIntervals"<< endl;
  if( ( _color<0 || _color>=ME::iSizeC ) || _type != ME::iLaser )
    {
      cout << "Validity Intervals: Only for LASER data" << endl;
      return 0;
    } 

  if( leaf->ig()<ME::iLMRegion )
    {
      cout << "Leaf must be at least with Side granularity" << endl;
      return 0;
    } 

  MEChannel* sideLeaf = leaf->getAncestor( ME::iLMRegion );
  if( _intervals[_color][_type].count( sideLeaf ) ) return _intervals[_color][_type][sideLeaf];
  buildMtqIntervals( createCor , leaf );
  
  assert( _intervals[_color][_type].count( sideLeaf )==1 );
  return _intervals[_color][_type][sideLeaf];
}

MEIntervals* 
MusEcal::tpIntervals( MEChannel *leaf )
{
  
  if(_debug) cout<< "Entering tpIntervals"<< endl;
  if( leaf->ig()<ME::iLMRegion )
    {
      cout << "Leaf must be at least with Side granularity" << endl;
      return 0;
    } 
  
  if( (  _type != ME::iTestPulse ) || ( _color!=0 ) )
    {
      if( _intervals[0][ME::iTestPulse].count(leaf )) return _intervals[0][ME::iTestPulse][leaf];
      else{
	cout << "TP Validity Intervals have not been set for LASER data" << endl;
	return 0;
      }
    } 
  
  if( _intervals[_color][_type].count(leaf )) return _intervals[_color][_type][leaf];
  buildTPIntervals(leaf);
  
  assert( _intervals[_color][_type].count(leaf)==1 );
  return _intervals[_color][_type][leaf];
}

std::vector< std::pair<ME::Time, ME::Time> >
MusEcal::tpFlagIntervals( MEChannel *leaf, int pnNum, double cutStep, double cutSlope )
{
  
  std::vector< std::pair<ME::Time, ME::Time> > myBadTPIntervals;
  
  if(_debug) cout<< "Entering tpFlagIntervals"<< endl;

  if( leaf->ig()<ME::iLMModule )
    {
      cout << "Leaf must be at least with module granularity" << endl;
      assert( leaf->ig()==ME::iLMModule );
    } 
  
  if(  _type != ME::iTestPulse  )
    {
      cout << "TP Validity Intervals have not been set for LASER data" << endl;
      assert( _type == ME::iTestPulse  );
    }
  
  MEVarVector* pnaVector = curMgr()->pnVector(leaf, pnNum);

  vector< bool  > flagPN; 
  vector< float > valPN;

  vector< ME::Time > time; 
  pnaVector->getTime( time );
  pnaVector->getValAndFlag( ME::iTPPN_MEAN, time, valPN, flagPN );

  int n=8;

  ME::Time time0=time[0];


  for (unsigned int i=0;i<valPN.size()-n;i++){

    std::pair<ME::Time, ME::Time> badInt;
    vector<float> delta8;
    vector<ME::Time> time8;
    double slopes=0;
    bool fourneg=true;

    for (int j=0;j<n;j++){
      int k=i+j;
      delta8.push_back( 2.0*(valPN[k+1]-valPN[k])/ (valPN[k+1]+valPN[k]) );
      time8.push_back(time[k]);
      if(j>0){
	slopes+=0.5*(valPN[k+1]-valPN[k])/(valPN[k+1]+valPN[k]);
	if(valPN[k+1]-valPN[k]>0) fourneg=false;
      }
    } 


    if( delta8[0] > cutStep && TMath::Abs(slopes)>cutSlope ){
      
      badInt.first=time[i];
      if(i+n<valPN.size()) badInt.second=time[i+n];
      else badInt.second=time[time.size()-1];      
      myBadTPIntervals.push_back(badInt);
      cout<< "First: "<<ME::timeDiff( badInt.first,time0 )<< "   Last: "<<ME::timeDiff( badInt.second,time0 )<< endl;
      i+=n-1;      
    }
    
  }  
  return myBadTPIntervals;
}

// void 
// MusEcal::setTPIntervals( MEChannel *leaf, MEIntervals* intervals ){
  
//   if(_type != ME::iLaser ) {
    
//     cout<< "setTPIntervals to be used only for LASER data"<< endl;
//     return;
//   } 
//   _intervals[0][ME::iTestPulse][leaf]=intervals;  
// }

// void 
// MusEcal::setLASERIntervals( MEChannel *leaf, MEIntervals* intervals, int color ){
  
//   if(_type != ME::iLaser ) {
    
//     cout<< "setTPIntervals to be used only for LASER data"<< endl;
//     return;
//   } 
//   _intervals[color][ME::iLaser][leaf]=intervals;  
// }

void
MusEcal::setMtqVar( int mtqVar0, int mtqVar1, int mtqLevel )
{
  if( mtqVar0==_mtqVar0 && mtqVar1==_mtqVar1 && mtqLevel==_mtqLevel ) return;
  bool ok=true;
  if( mtqVar0<0  || mtqVar0>=ME::iSizeMTQ )  return;
  //if( iGVar[mtqVar0]!=ME::iLMRegion ) ok = false; // FIXME
  if( !ok )
    {
      cout << "Warning in setMTQVar: the first variable is not a valid MATACQ variable, mtqVar0=" << mtqVar0 << endl;
      ok = true;
    }
  _mtqVar0   = mtqVar0;
  _mtqVar1   = mtqVar1;
  _mtqLevel  = mtqLevel;
  if( _mtqVar1<0  || _mtqVar1>=ME::iSizeMTQ )          _mtqVar1 = -1;
  
  _mtqTwoVar = true;
  _tpTwoVar = true;
  if( _mtqVar1==-1 ) _mtqTwoVar = false;
  if( _tpVar1==-1 )_tpTwoVar = false;

  if( _mtqLevel<0 ) _mtqLevel  = MEIntervals::level5;

  // delete the intervals at the Side level
  
  for( int color=0; color<ME::iSizeC; color++ ) 
    {
      map< MEChannel*, MEIntervals* >::iterator it;
      for( it=_intervals[color][_type].begin(); it!=_intervals[color][_type].end(); it++ )
	{
	  MEChannel* leaf_ = it->first;
	  if( leaf_->ig() == ME::iLMRegion ) 
	    {
	      MEIntervals* intervals_ =  it->second;
	      _intervals[color][_type].erase( leaf_ );
	      delete intervals_;
	    }
	}
    }
}

void
MusEcal::setTPVar( int tpVar0, int tpVar1, int tpLevel )
{
  if( tpVar0==_tpVar0 && tpVar1==_tpVar1 && tpLevel==_tpLevel ) return;
  bool ok=true;
  
  if( tpVar0<0  || tpVar0>=ME::iSizeTPAPD )  return;
  //if( iGVar[tpVar0]!=ME::iLMRegion ) ok = false; // FIXME
  if( !ok )
    {
      cout << "Warning in setTPVar: the first variable is not a valid TP variable, tpVar0=" << tpVar0 << endl;
      ok = true;
    }
  _tpVar0   = tpVar0;
  _tpVar1   = tpVar1;
  _tpLevel  = tpLevel;
  if( _tpVar1<0  || _tpVar1>=ME::iSizeTPAPD )_tpVar1 = -1;

  _tpTwoVar = true;
  if( _tpVar1==-1 ) _tpTwoVar = false;
  
  if( _tpLevel<0 ) _tpLevel  = MEIntervals::level5;
  
  // delete the intervals at the Xtal level
  
  int color=0;
  map< MEChannel*, MEIntervals* >::iterator it;
  for( it=_intervals[color][_type].begin(); it!=_intervals[color][_type].end(); it++ )
    {
      MEChannel* leaf_ = it->first;
      if( leaf_->ig() == ME::iCrystal ) 
	{
	  MEIntervals* intervals_ =  it->second;
	  _intervals[color][_type].erase( leaf_ );
	  delete intervals_;
	}
    }
}

void
MusEcal::setPNVar( int pnVar0, int pnVar1, int pnLevel )
{
  if( pnVar0==_pnVar0 && pnVar1==_pnVar1 && pnLevel==_pnLevel ) return;
  bool ok=true;
  
  if( pnVar0<0  || pnVar0>=ME::iSizePN )  return;

  if( !ok )
    {
      cout << "Warning in setPNVar: the first variable is not a valid PN variable, pnVar0=" << pnVar0 << endl;
      ok = true;
    }
  _pnVar0   = pnVar0;
  _pnVar1   = pnVar1;
  _pnLevel  = pnLevel;
  if( _pnVar1<0  || _pnVar1>=ME::iSizePN)_pnVar1 = -1;

  _pnTwoVar = true;
  if( _pnVar1==-1 ) _pnTwoVar = false;
  
  if( _pnLevel<0 ) _pnLevel  = MEIntervals::level5;
  
  // delete the intervals at the Xtal level
  
  int color=0;
  map< MEChannel*, MEIntervals* >::iterator it;
  for( it=_intervals[color][_type].begin(); it!=_intervals[color][_type].end(); it++ )
    {
      MEChannel* leaf_ = it->first;
      if( leaf_->ig() == ME::iCrystal ) 
	{
	  MEIntervals* intervals_ =  it->second;
	  _intervals[color][_type].erase( leaf_ );
	  delete intervals_;
	}
    }
}

void
MusEcal::buildPNIntervals( std::vector<int>& iapdopn , MEChannel *leaf )
{
  if(_debug) cout<< "Entering buildPNIntervals"<< endl;

  MEChannel* modLeaf = leaf->getAncestor( ME::iLMModule );
  MEVarVector* pnaVector_ = curMgr()->pnVector(modLeaf,0);
  MEVarVector* pnbVector_ = curMgr()->pnVector(modLeaf,1);

  ME::Time firstKey = curMgr()->firstKey();
  ME::Time lastKey  = curMgr()->lastKey();

  // Build eventually pn intervals with pn/pn
  // =========================================
  //MEIntervals* pnintervals_ = new MEIntervals( pnaVector_ , pnaVector_ ,  firstKey, lastKey, _pnVar0, _pnVar1 ); 
  //pnintervals_->splitIntervals( MEIntervals::threshold[ _pnLevel ] );
  
  METimeInterval *pnintervals_= new METimeInterval(firstKey, lastKey );
  
  // Do here some more work on variable determination etc
  //======================================================
  list<ME::Time> times;
  vector<int> vars;
  vector< ME::Time > time;
  MEVarVector* apdVector_ = curMgr()->apdVector(leaf);
  apdVector_->getTime( time );

  vector< float > valPNoPN;
  vector< bool > flagPNoPN;
  vector< float > valPNA;
  vector< bool  > flagPNA; 
  vector< float > valPNB;
  vector< bool  > flagPNB;
  vector< float > rmsPNA;
  vector< float > rmsPNB;
    
  pnaVector_->getValAndFlag( ME::iPN_RMS, time, rmsPNA, flagPNA );
  pnbVector_->getValAndFlag( ME::iPN_RMS, time, rmsPNB, flagPNB );
  pnaVector_->getValAndFlag( ME::iPN_MEAN, time, valPNA, flagPNA );
   pnbVector_->getValAndFlag( ME::iPN_MEAN, time, valPNB, flagPNB );
  pnaVector_->getValAndFlag( ME::iPNA_OVER_PNB_MEAN, time, valPNoPN, flagPNoPN );

  assert(rmsPNA.size()==rmsPNB.size());
  assert(valPNA.size()==rmsPNA.size());

  int defVar;
  int curVar;
  int lastVar=-999;

  double PNoPNMin=0.8;
  double PNoPNMax=1.2;
  double RMSPNMin=0.01;
  double RMSPNMax=0.06;

  // push_back first time 
  //if(time.size()>0) times.push_back(time[0]);

  for (unsigned int j=0;j<valPNA.size();j++){
    
    double rmsArel=0.0;
    double rmsBrel=0.0;
    if(valPNA[j]>0.0) rmsArel=rmsPNA[j]/valPNA[j];
    if(valPNB[j]>0.0) rmsBrel=rmsPNB[j]/valPNB[j];
    
    bool isPNoPNOK = ( valPNoPN[j]<PNoPNMax && valPNoPN[j]>PNoPNMin );
    bool isPNAOK   = ( rmsArel<RMSPNMax && rmsArel>RMSPNMin   );
    bool isPNBOK   = ( rmsBrel<RMSPNMax  && rmsBrel>RMSPNMin );
    defVar=ME::iMID_MEAN;

    if(flagPNA[j] && flagPNB[j] && isPNoPNOK && isPNAOK && isPNBOK )
      curVar=ME::iMID_MEAN;
    else if( flagPNA[j] && isPNAOK && (!isPNBOK || !flagPNB[j]))
      curVar=ME::iMIDA_MEAN;
    else if( flagPNB[j] && isPNBOK && (!isPNAOK || !flagPNA[j]))
      curVar=ME::iMIDB_MEAN;
    else curVar=defVar;

    if(lastVar!=curVar){
      if(_debug)
	cout<< "New PN Interval: "<<j<<" "<<time[j]<<" "<<firstKey<<" "<< lastVar<<" "<<curVar<<endl;
      if(_debug) cout << "rmsA:"<<rmsArel<<" rmsB:"<<rmsBrel<<" pnopn:"<<valPNoPN[j]<< endl;
      if(_debug) cout<< "isPNoPNOK:"<< isPNoPNOK <<" isPNAOK:"<< isPNAOK<<" isPNBOK:"<< 
		   isPNBOK<<" flagA:"<<flagPNA[j]<<" flagB:"<<flagPNB[j]<<endl;
      
      
      // push_back next time 
      times.push_back(time[j]);
      vars.push_back(curVar);
    }
    lastVar=curVar;
  }
  
  if(_debug) cout<< "buildPNIntervals times size:"<<times.size()<<" vars size:"<<vars.size()<< endl;
  
  // for testing:

  pnintervals_->split(times);
  iapdopn=vars;
 
  _pnvarmap[_color][modLeaf] = vars;
  _pnintervals[_color][modLeaf] = pnintervals_;
 
}

void
MusEcal::buildMtqIntervals( bool createCor , MEChannel *leaf )
{
  if(_debug) cout<< "Entering buildMtqIntervals"<< endl;
  MEChannel* sideLeaf = leaf->getAncestor( ME::iLMRegion );
  MEVarVector* mtqVector_ = curMgr()->mtqVector(sideLeaf); 

  ME::Time firstKey = curMgr()->firstKey();
  ME::Time lastKey  =  curMgr()->lastKey();

  MEIntervals* intervals_ = new MEIntervals( mtqVector_, mtqVector_, firstKey, lastKey, _mtqVar0, _mtqVar1 ); 

  intervals_->splitIntervals( MEIntervals::threshold[ _mtqLevel ] );

  _intervals[_color][_type][sideLeaf] = intervals_;
 
  //  if(createCor) createCorrectors(leaf);
  
}

void
MusEcal::buildTPIntervals(MEChannel *leaf)
{
  if(_debug) cout<< "Entering buildTPIntervals"<< endl;
 
  MEVarVector* tpVector_ = curMgr()->apdVector(leaf); 
  
  ME::Time firstKey = curMgr()->firstKey();
  ME::Time lastKey  = curMgr()->lastKey();
  
  if(_debug) cout<< "First and last keys: "<< firstKey<<" "<<lastKey<<endl;
  int color=0;
  MEIntervals* intervals_ = new MEIntervals( tpVector_, tpVector_,  firstKey, lastKey, _tpVar0, -1); 
  
  intervals_->splitIntervals( MEIntervals::threshold[ _tpLevel ] );
  
  _intervals[color][_type][leaf] = intervals_;
  
}


// void
// MusEcal::createCorrectors(MEChannel *leaf)
// {
//   if(_debug) cout<< "Entering createCorrectors"<< endl;
 
//   int type = _type;
//   int var0 = _corVar0;
//   int var1 = _corVar1;
//   int zoom0 = _corZoom0;
//   int zoom1 = _corZoom1;
//   unsigned fitDegree = _corFitDegree;
//   double x0 = _corX0;
//   double y0 = _corY0;
//   vector< double >& beta = _corBeta; 

//   if( leaf->ig()<ME::iLMModule )
//     {
//       cout << "Please select a leaf at least with Module granularity" << endl;
//       return;
//     } 

//   MEChannel* sideLeaf = leaf->getAncestor( ME::iLMRegion );
//   MEVarVector* mtqVector_ = curMgr()->mtqVector(sideLeaf); 
//   MEVarVector* midVector_ = midVector(leaf); 

//   // get the intervals
//   MEIntervals* intervals_    = mtqIntervals( false, leaf );

//   // get the top interval
//   METimeInterval* topInterval = intervals_->topInterval();

//   int varIndex[2] = { var0, var1 };
//   int varZoom[2] = { zoom0, zoom1 };

//   // clean-up the correctors
//   //      if( _debug ) 
//   if( _debug ) cout << "Cleanup correctors " << endl;
//   MECorrector2Var::cleanup();

//   if( _debug ) cout << "Create Correctors for Leaf " << leaf->oneWord() 
//        << " and Variables (" << var0 << "," << var1 << ")" <<  endl;

//   // loop on the intervals at the 
//   METimeInterval* ki(0);

//   if( _debug ) cout << "Create corrector for top interval " << endl;
//   MECorrector2Var* cor_(0); 
//   cor_ = new MECorrector2Var( mtqVector_ , midVector_, topInterval, 
// 			      type, varIndex[0], varIndex[1], varZoom[0], varZoom[1], 
// 			      fitDegree, x0, y0 );
//   assert( cor_!=0 );
//   if( beta.size()==fitDegree+1 )
//     {
//       cor_->setBetaParams( beta );
//     }

//   //  unsigned nlevelmax = MEIntervals::nlevel;
//   unsigned nlevelmax = 1;
//   unsigned ilevel= nlevelmax;
//   //  for( unsigned ilevel=nlevelmax; ilevel!=0; ilevel-- )
//   //    {
//   if(_debug)
//     {
//       cout << "Intervals at level " << ilevel << endl;
//       topInterval->print( ilevel );
//     }
//   int interval_(0);
//   for( ki=topInterval->first(ilevel); ki!=0; ki=ki->next() )
//     {
//       interval_++;
//       if( _debug )
// 	{
// 	  cout << "--> interval "  << interval_ <<  endl; 
// 	  cout << ki->inBrackets();
// 	  cout << " next=" << ki->next();
// 	  cout << " previous=" << ki->previous();
// 	  cout << " firstIn=" << ki->firstIn();
// 	  cout << " lastIn=" << ki->lastIn();
// 	  cout << endl;
// 	}
      
//       assert( MECorrector2Var::corMap.count( ki )==0 );
//       cor_ = new MECorrector2Var( mtqVector_, midVector_, ki, 
// 				  type, varIndex[0], varIndex[1], varZoom[0],
// 				  varZoom[1], fitDegree, x0, y0);
//       assert( cor_!=0 );
//       if( beta.size()==fitDegree+1 )
// 	{
// 	  cor_->setBetaParams( beta );
// 	}
//       if( _debug ) 
// 	cor_->print();
//     }
//   delete cor_;
//   delete intervals_;
// }

void
MusEcal::fillMIDMaps()
{
  while( !_midMap.empty() ) 
    { 
      delete _midMap.begin()->second;
      _midMap.erase( _midMap.begin() );
    }

  vector< MEChannel* > listOfChan_;
  listOfChan_.clear();
  curMgr()->tree()->getListOfDescendants( ME::iCrystal, listOfChan_ );

  unsigned int size_(0);

  if( _type==ME::iLaser )
    {
      size_=ME::iSizeMID;
    }
  else{
    return;
  }
  
  cout << "Filling MID Maps (number of C=" << listOfChan_.size() << ")" << endl;
  
  for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
    {
      MEChannel* leaf_ = listOfChan_[ichan];  
      
      if( _midMap.count(leaf_)==0 )
	{
	  _midMap[leaf_]=fillMIDVector(leaf_);
	}
    }
  cout << "...done." << endl;

  //
  // At higher levels
  
  cout << "Filling MID Maps for other granularities" << endl;
  MERunManager* mgr_ = runMgr();
  
  for( MusEcal::RunIterator p=mgr_->begin(); p!=mgr_->end(); ++p )
    {
      MERun* run_ = p->second;
      ME::Time time = run_->time();
      
      for( int ig=ME::iSuperCrystal; ig>=ME::iLMRegion; ig-- )
	{
	  listOfChan_.clear();
	  if( ig==ME::iLMRegion ) listOfChan_.push_back( mgr_->tree() );
	  else  mgr_->tree()->getListOfDescendants( ig, listOfChan_ );
	  
	  for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
	    {
	      MEChannel* leaf_ = listOfChan_[ichan];
	      if( _midMap.count(leaf_)==0 )
		{
		  _midMap[leaf_] = new MEVarVector( size_ ); 
		}
	      MEVarVector* varVector_ = _midMap[leaf_];

	      varVector_->addTime( time );
	      for( unsigned ii=0; ii<=ME::iSizeMID; ii++ )
		{
		  float val=0;
		  float n=0;
		  // loop on daughters
		  for( unsigned int idau=0; idau<leaf_->n(); idau++ )
		    {
		      float val_(0);
		      bool flag_=true;
		      
		      assert( _midMap[leaf_->d(idau)]
			      ->getValByTime( time, ii, val_, flag_ ) );
		      
		      if( val_>0 )
			{
			  n++;
			  val+=val_;
			}
		    }
		  if( n!=0 ) val/=n; 
		  varVector_->setVal( time, ii, val );
		}
	    }
	}
    }
  
  cout << "...done." << endl;
}

MEVarVector*
MusEcal::midVector( MEChannel* leaf )
{
  if( _midMap.count( leaf ) !=0 ) return _midMap[leaf];
  else
    return 0;
} 

 MEVarVector* 
 MusEcal::fillMIDVector(MEChannel* leaf_){

   if(_debug) cout<<" Entering fillMIDVector "<< leaf_->ig()<<endl;
   assert(leaf_->ig()==ME::iCrystal);

   unsigned int size_(0);
   if( _type==ME::iLaser )
     {
       size_=ME::iSizeMID;
     }
   MEVarVector* midvec= new MEVarVector( size_ );
   MEVarVector* apdVector_ = curMgr()->apdVector(leaf_);
   MEChannel* pnleaf_=leaf_;

   if(leaf_->ig()>ME::iLMModule){
     while( pnleaf_->ig() != ME::iLMModule){
       pnleaf_=pnleaf_->m();
     }
   }
   
   MEVarVector* pnaVector_ = curMgr()->pnVector(pnleaf_,0);
   MEVarVector* pnbVector_ = curMgr()->pnVector(pnleaf_,1);


   // retrive mem and pn for pn correction
   
   int iPNA, iPNB;
   int memA, memB;
   
   int ix=leaf_->ix();
   int iy=leaf_->iy();
   
   int reg= ME::ecalRegion( _lmr );
   int iz=1;
   
   std::pair< int, int > dccSide=ME::dccAndSide( _lmr );
   
   if( reg==ME::iEEM ) iz=-1;
   
   pair< int, int > mems=ME::memFromLmr(_lmr);
   memA=mems.first;
   memB=mems.second;
   
   std::pair< int, int > pns;
   if( reg==ME::iEEM || reg==ME::iEEP){
     int iX = (ix-1)/5+1;
     int iY = (iy-1)/5+1;
     int dee=MEEEGeom::dee(iX,iY,iz);
     int lmmod=MEEEGeom::lmmod(iX,iY);
     pns=MEEEGeom::pn(dee,lmmod);
     
   }else{
     //pair< int, int > gxy=MEEBGeom::globalCoord( MEEBGeom::smFromDcc(dccSide.first), ix, iy );
     //ixglob=gxy.first;
     //iyglob=gxy.second;
     int lmmod=MEEBGeom::lmmod(ix,iy);
     pns=MEEBGeom::pn(lmmod,dccSide.first);
   }
   
   iPNA=pns.first;
   iPNB=pns.second;
   int PNgain=0;
   

   vector< float > valtake;
   vector< bool  > flagtake;
   vector< float  > nevttake;
   vector< float  > rmstake;

   vector< float > valinit[3];
   vector< bool  > flaginit[3];
   vector< float > nevtinit[3];
   vector< float > rmsinit[3];

   vector< float > shapeCorVal;
   vector< bool >   shapeCorFlag;

   vector< float > shapeCorPNAVal;
   vector< bool >   shapeCorPNAFlag;

   vector< float > shapeCorPNBVal;
   vector< bool >   shapeCorPNBFlag;

   vector< float > PNAVal;
   vector< bool >  PNAFlag;

   vector< float > PNBVal;
   vector< bool >  PNBFlag;

   
   vector< bool  > flag_;
   int varMean[3];
   int varRMS[3];
   int varNevt[3];
   varMean[0]=ME::iAPD_OVER_PN_MEAN;
   varMean[1]=ME::iAPD_OVER_PNA_MEAN;
   varMean[2]=ME::iAPD_OVER_PNB_MEAN;
   varNevt[0]=ME::iAPD_OVER_PN_NEVT;
   varNevt[1]=ME::iAPD_OVER_PNA_NEVT;
   varNevt[2]=ME::iAPD_OVER_PNB_NEVT;
   varRMS[0]=ME::iAPD_OVER_PN_RMS;
   varRMS[1]=ME::iAPD_OVER_PNA_RMS;
   varRMS[2]=ME::iAPD_OVER_PNB_RMS;
   
   int midVarMean[3];
   int midVarRMS[3];
   int midVarNevt[3];
   midVarMean[0]=ME::iMID_MEAN;
   midVarMean[1]=ME::iMIDA_MEAN;
   midVarMean[2]=ME::iMIDB_MEAN;
   midVarRMS[0]=ME::iMID_RMS;
   midVarRMS[1]=ME::iMIDA_RMS;
   midVarRMS[2]=ME::iMIDB_RMS;
   midVarNevt[0]=ME::iMID_NEVT;
   midVarNevt[1]=ME::iMIDA_NEVT;
   midVarNevt[2]=ME::iMIDB_NEVT;

   int tmpVarMean[3];
   int tmpVarRMS[3];
   int tmpVarNevt[3];
   tmpVarMean[0]=ME::iAPD_OVER_PNTMPCOR_MEAN;
   tmpVarMean[1]=ME::iAPD_OVER_PNATMPCOR_MEAN;
   tmpVarMean[2]=ME::iAPD_OVER_PNBTMPCOR_MEAN;
   tmpVarRMS[0]=ME::iAPD_OVER_PNTMPCOR_RMS;
   tmpVarRMS[1]=ME::iAPD_OVER_PNATMPCOR_RMS;
   tmpVarRMS[2]=ME::iAPD_OVER_PNBTMPCOR_RMS;
   tmpVarNevt[0]=ME::iAPD_OVER_PNTMPCOR_NEVT;
   tmpVarNevt[1]=ME::iAPD_OVER_PNATMPCOR_NEVT;
   tmpVarNevt[2]=ME::iAPD_OVER_PNBTMPCOR_NEVT;

   vector< ME::Time > time; 
   if(apdVector_!=0){
     apdVector_->getTime( time );
     apdVector_->getValAndFlag(ME::iAPD_SHAPE_COR, time, shapeCorVal, shapeCorFlag );
   }else{
     return midvec;
   }
   if(pnaVector_!=0){
     pnaVector_->getValAndFlag(ME::iPN_SHAPE_COR, time, shapeCorPNAVal, shapeCorPNAFlag );
     pnaVector_->getValAndFlag(ME::iPN_MEAN, time, PNAVal, PNAFlag );
   }
   if(pnbVector_!=0){
     pnbVector_->getValAndFlag(ME::iPN_SHAPE_COR, time, shapeCorPNBVal, shapeCorPNBFlag );
     pnbVector_->getValAndFlag(ME::iPN_MEAN, time, PNBVal, PNBFlag );
   }
   for( int ivar=0;ivar<3;ivar++){  

     valtake.clear();
     rmstake.clear();
     nevttake.clear();
     flagtake.clear();
     
     apdVector_->getValAndFlag( varMean[ivar], time, valtake, flagtake );
     apdVector_->getValAndFlag( varRMS[ivar],  time, rmstake,  flag_ );
     apdVector_->getValAndFlag( varNevt[ivar], time, nevttake, flag_ );
     
     valinit[ivar]=valtake;
     rmsinit[ivar]=rmstake;
     nevtinit[ivar]=nevttake;
     flaginit[ivar]=flagtake;
   }
   
   unsigned int nrunmax = valinit[0].size();
   
   assert( valinit[0].size()==valinit[1].size());
   assert( valinit[1].size()==valinit[2].size());
   assert( valinit[1].size()==valinit[2].size());
   assert( shapeCorVal.size()==valinit[2].size());

   double shapeCorMin=0.7;
   double shapeCorPNMin=0.9;
   
   if( ME::ecalRegion( _lmr )!=ME::iEBM && ME::ecalRegion( _lmr )!=ME::iEBP ){
     shapeCorMin=0.5; 
     shapeCorPNMin=0.7;
   }
 
   for (unsigned int irun=0;irun<nrunmax;irun++){
     
     float PNAcor=PNAVal[irun];
     float PNBcor=PNBVal[irun];
     float PNcor; float PNval;
     
     PNAcor=_calibData->getPNCorrected(PNAVal[irun],iPNA,PNgain,memA);
     PNBcor=_calibData->getPNCorrected(PNBVal[irun],iPNB,PNgain,memB);
     
     if (PNAVal[irun]<50 && PNBVal[irun]>50) {
       PNcor=PNBcor;
       PNval=PNBVal[irun];
     }else if (PNBVal[irun]<50 && PNAVal[irun]>50){
       PNcor=PNAcor;
       PNval=PNAVal[irun];
     }else{
       PNcor=0.5*(PNBcor+PNAcor);
       PNval=0.5*(PNAVal[irun]+PNBVal[irun]);
     }
     
     for( int ivar=0;ivar<3;ivar++){  
       
       float corVal=0.;
       float rmsVal=0.;
       float corTmp=0.;
       float rmsTmp=0.;

       float pnCorrectionFactor=1.0;
       
       bool thisflag=flaginit[ivar][irun];
       bool thisflagtmp=flaginit[ivar][irun];
       
       
       // Calculate linearity correction:
       //=============================
       if(ivar==0 && PNcor>0. ){
	 pnCorrectionFactor=PNval/PNcor;
	 if(!PNAFlag[irun] || !PNBFlag[irun] ) thisflag=false;
       }else if(ivar==1 && PNAcor>0.){
	 pnCorrectionFactor=PNAVal[irun]/PNAcor;
	 if(!PNAFlag[irun]  ) thisflag=false;
       }else if(ivar==2 && PNBcor>0.){
	 pnCorrectionFactor=PNBVal[irun]/PNBcor;
	 if(!PNBFlag[irun]  ) thisflag=false;
       }

       thisflagtmp=thisflag;
       corTmp = valinit[ivar][irun]*pnCorrectionFactor;
       rmsTmp = rmsinit[ivar][irun]*pnCorrectionFactor;   
       
       // Apply shape correction and linearity correction:
       //================================================
       
       if(shapeCorVal[irun]>shapeCorMin){
	 
	 corVal = (valinit[ivar][irun]/shapeCorVal[irun])*pnCorrectionFactor;
	 rmsVal = (rmsinit[ivar][irun]/shapeCorVal[irun])*pnCorrectionFactor;  
	 
	 if(ivar==0 && shapeCorPNAVal[irun]>shapeCorPNMin && shapeCorPNBVal[irun]>shapeCorPNMin && pnaVector_!=0  && pnbVector_!=0 ){
	   
	   corVal*=0.5*(shapeCorPNAVal[irun]+shapeCorPNBVal[irun]);
 	   rmsVal*=0.5*(shapeCorPNAVal[irun]+shapeCorPNBVal[irun]);
	   
	 }else if(ivar==1 && pnaVector_!=0 && shapeCorPNAVal[irun]>shapeCorPNMin){
	   
	   corVal*=shapeCorPNAVal[irun];
	   rmsVal*=shapeCorPNAVal[irun];
	   
	 }else if(ivar==2 && pnbVector_!=0 && shapeCorPNBVal[irun]>shapeCorPNMin){
	   
	   corVal*=shapeCorPNBVal[irun];
	   rmsVal*=shapeCorPNBVal[irun];
	   
	 }else{
	   
	   thisflag=false;
	   
	   if(pnaVector_!=0 && pnbVector_!=0){
	     if(shapeCorPNAVal[irun]>0.1 && shapeCorPNBVal[irun]<0.1 ){
	       
	       corVal*=shapeCorPNAVal[irun];
	       rmsVal*=shapeCorPNAVal[irun];
	       
	     }else if(shapeCorPNAVal[irun]<0.1 && shapeCorPNBVal[irun]>0.1){
	       
	       corVal*=shapeCorPNBVal[irun];
	       rmsVal*=shapeCorPNBVal[irun];
	       
	     }	
	   }   
	 }
       }else{
	 thisflag=false;
       }
       
       midvec->setValAndFlag(time[irun],tmpVarMean[ivar], corTmp, thisflagtmp, true);
       midvec->setValAndFlag(time[irun],tmpVarNevt[ivar], nevtinit[ivar][irun], thisflagtmp, true);
       midvec->setValAndFlag(time[irun],tmpVarRMS[ivar], rmsTmp, thisflagtmp, true);
       
       midvec->setValAndFlag(time[irun],midVarMean[ivar],corVal,thisflag,true);
       midvec->setValAndFlag(time[irun],midVarNevt[ivar],nevtinit[ivar][irun],thisflag,true);
       midvec->setValAndFlag(time[irun],midVarRMS[ivar], rmsVal,thisflag,true);
     }
   }
   return midvec;
 }
     
// MEVarVector* 
// MusEcal::fillMIDVector(MEChannel* leaf_)
// {
//   if(_debug) cout<<" Entering fillMIDVector "<< leaf_->ig()<<endl;
//   assert(leaf_->ig()==ME::iCrystal);
  
//   MEChannel* pnleaf_=leaf_;
//   MEChannel* mtqleaf_=leaf_;

//   if(leaf_->ig()>ME::iLMModule){
//     while( pnleaf_->ig() != ME::iLMModule){
//       pnleaf_=pnleaf_->m();
//     }
//   }
//   if(leaf_->ig()>ME::iLMRegion){
//     while( mtqleaf_->ig() != ME::iLMRegion){
//       mtqleaf_=mtqleaf_->m();
//     }
//   }

//   unsigned int size_(0);
//   if( _type==ME::iLaser )
//     {
//       size_=ME::iSizeMID;
//     }

//   MEVarVector* midvec= new MEVarVector( size_ );
    
//   int corVar0 = ME::iMTQ_FWHM; // FIXME change this to be a variable

//   int corVar1[3];
//   corVar1[0]=ME::iMID_MEAN;
//   corVar1[1]=ME::iMIDA_MEAN;
//   corVar1[2]=ME::iMIDB_MEAN;
  
//   int midvarmean[3];
//   int midvarrms[3];
//   int midvarnevt[3];
//   midvarmean[0]=ME::iMID_MEAN;
//   midvarmean[1]=ME::iMIDA_MEAN;
//   midvarmean[2]=ME::iMIDB_MEAN;
//   midvarrms[0]=ME::iMID_RMS;
//   midvarrms[1]=ME::iMIDA_RMS;
//   midvarrms[2]=ME::iMIDB_RMS;
//   midvarnevt[0]=ME::iMID_NEVT;
//   midvarnevt[1]=ME::iMIDA_NEVT;
//   midvarnevt[2]=ME::iMIDB_NEVT;
//   int midvarcormean[3];
//   int midvarcorrms[3];
//   int midvarcornevt[3];
//   midvarcormean[0]=ME::iMID_COR_MEAN;
//   midvarcormean[1]=ME::iMIDA_COR_MEAN;
//   midvarcormean[2]=ME::iMIDB_COR_MEAN;
//   midvarcorrms[0]=ME::iMID_COR_RMS;
//   midvarcorrms[1]=ME::iMIDA_COR_RMS;
//   midvarcorrms[2]=ME::iMIDB_COR_RMS;
//   midvarcornevt[0]=ME::iMID_COR_NEVT;
//   midvarcornevt[1]=ME::iMIDA_COR_NEVT;
//   midvarcornevt[2]=ME::iMIDB_COR_NEVT;

//   vector< float > valMID[3];
//   vector< float > valCorMID[3];
//   vector< bool  > flagMID[3];
//   vector< bool  > flagCorMID[3];
//   vector< float > rmsMID[3];
//   vector< float > rmsCorMID[3];
//   vector< ME::Time > time; 

//   MEVarVector* apdVector_ = curMgr()->apdVector(leaf_);
//   MEVarVector* pnaVector_ = curMgr()->pnVector(pnleaf_,0);
//   MEVarVector* pnbVector_ = curMgr()->pnVector(pnleaf_,1);
//   MEVarVector* mtqVector_ = curMgr()->mtqVector(mtqleaf_); 

//   apdVector_->getTime( time );

//   // APDPN Stuff

//   vector< float > valtake;
//   vector< bool  > flagtake;
//   vector< float  > nevttake;
//   vector< float  > rmstake;
//   vector< float > valinit[3];
//   vector< float > rmsinit[3];
//   vector< bool  > flaginit[3];
//   vector< float > nevtinit[3];
//   vector< float > valpncor[3];
//   vector< float > rmspncor[3];
//   vector< bool  > flag_;
//   int varMean[3];
//   int varRMS[3];
//   int varNevt[3];
//   varMean[0]=ME::iAPD_OVER_PN_MEAN;
//   varMean[1]=ME::iAPD_OVER_PNA_MEAN;
//   varMean[2]=ME::iAPD_OVER_PNB_MEAN;
//   varNevt[0]=ME::iAPD_OVER_PN_NEVT;
//   varNevt[1]=ME::iAPD_OVER_PNA_NEVT;
//   varNevt[2]=ME::iAPD_OVER_PNB_NEVT;
//   varRMS[0]=ME::iAPD_OVER_PN_RMS;
//   varRMS[1]=ME::iAPD_OVER_PNA_RMS;
//   varRMS[2]=ME::iAPD_OVER_PNB_RMS;

//   // PN Stuff

//   vector< float > valPNA;
//   vector< bool  > flagPNA; 
//   vector< float > valPNB;
//   vector< bool  > flagPNB;
  
//   // Matacq Stuff

//   vector< float > valMatacq;
//   vector< bool  > flagMatacq;
  
//   pnaVector_->getValAndFlag( ME::iPN_MEAN, time, valtake, flagtake );
//   valPNA  = valtake; 
//   flagPNA = flagtake;

//   valtake.clear();
//   flagtake.clear();
//   pnbVector_->getValAndFlag( ME::iPN_MEAN, time, valtake, flagtake );
//   valPNB  = valtake; 
//   flagPNB = flagtake;
  
//   double corr[3];
//   double normAPDPN[3];
//   int cAPDPN[3];
//   double normMatacq[3];
//   int cMatacq[3];
//   double normMID[3];

//   for( int ivar=0;ivar<3;ivar++){  
//     valtake.clear();
//     rmstake.clear();
//     nevttake.clear();
//     flagtake.clear();

//     corr[ivar]=1.0;
//     normAPDPN[ivar]=0.0;
//     cAPDPN[ivar]=0; 
//     normMatacq[ivar]=0;
//     cMatacq[ivar]=0;
//     normMID[ivar]=1.0;
//     apdVector_->getValAndFlag( varMean[ivar], time, valtake, flagtake );
//     apdVector_->getValAndFlag( varRMS[ivar],  time, rmstake,  flag_ );
//     apdVector_->getValAndFlag( varNevt[ivar], time, nevttake, flag_ );

//     valinit[ivar]=valtake;
//     rmsinit[ivar]=rmstake;
//     nevtinit[ivar]=nevttake;
//     flaginit[ivar]=flagtake;
//   }
  
//   mtqVector_->getValAndFlag( corVar0 , time, valMatacq, flagMatacq );
  

//   // Apply PN linearity correction
  
//   unsigned int nrunmax = valinit[0].size();
 
//   assert( valinit[0].size()==valinit[1].size());
//   assert( valinit[1].size()==valinit[2].size());
//   assert( valinit[1].size()==valinit[2].size());
//   assert( valinit[0].size()==valPNA.size());
//   assert( valinit[0].size()==valPNB.size());
//   assert( valinit[0].size()==valMatacq.size());
  
//   for (unsigned int irun=0;irun<nrunmax;irun++){
    
//     corr[1]=_pnCorrector->getPNCorrectionFactor(valPNA[irun],0);
//     corr[2]=_pnCorrector->getPNCorrectionFactor(valPNB[irun],0);
    
    
//     if ( ( valPNA[irun]<10 && valPNB[irun]>10) || (!flagPNA[irun] && flagPNB[irun] )) {
//       corr[0]=corr[2];
//     }else if ( (valPNB[irun]<10 && valPNA[irun]>10) || (!flagPNB[irun] && flagPNA[irun])){
//       corr[0]=corr[1];
//     }else if(!flagPNB[irun] &&!flagPNA[irun] ){
//       corr[0]=1.0;
//     }else {
//       corr[0]=(corr[1]*valPNA[irun]+corr[2]*valPNB[irun])/(valPNA[irun]+valPNB[irun]);
//     }
    
    
//     for( int ivar=0;ivar<3;ivar++){  
      
//       if(corr[ivar]!=0){
// 	valpncor[ivar].push_back(valinit[ivar][irun]/corr[ivar]);
// 	rmspncor[ivar].push_back(rmsinit[ivar][irun]/corr[ivar]);
//       }
//       else{
// 	valpncor[ivar].push_back(valinit[ivar][irun]);
// 	rmspncor[ivar].push_back(rmsinit[ivar][irun]);
//       }
//       if(flaginit[ivar][irun]){  
// 	normAPDPN[ivar]+=valinit[ivar][irun]/corr[ivar];
// 	cAPDPN[ivar]++;
	
// 	if(flagMatacq[irun]){
// 	  normMatacq[ivar]+=valMatacq[irun];
// 	  cMatacq[ivar]++;
	  
// 	} 
//       }
      
//       if (flaginit[ivar][irun] && corr[ivar]!=0 ) flagMID[ivar].push_back(true);
//       else flagMID[ivar].push_back(false);
//     }
//   }
  
//   assert( valinit[0].size()==valpncor[0].size());
//   assert( valinit[0].size()==rmspncor[0].size());
  

//   for( int ivar=0;ivar<3;ivar++){  

//     if( cAPDPN[ivar]==0){
//       for (unsigned int irun=0;irun<nrunmax;irun++){
// 	midvec->setValAndFlag(time[irun],midvarmean[ivar],valinit[ivar][irun],false,true);
// 	midvec->setValAndFlag(time[irun],midvarnevt[ivar],nevtinit[ivar][irun],false,true);
// 	midvec->setValAndFlag(time[irun],midvarrms[ivar],rmsinit[ivar][irun],false,true);
//       }
//     }else{
      
//       assert(cAPDPN[ivar]);
//       normAPDPN[ivar]/=double(cAPDPN[ivar]);
      
//       for (unsigned int irun=0;irun<nrunmax;irun++){
	
// 	double myval=valpncor[ivar][irun];
	
// 	valMID[ivar].push_back(myval);
// 	rmsMID[ivar].push_back(rmspncor[ivar][irun]);   
	
// 	midvec->setValAndFlag(time[irun],midvarmean[ivar],valMID[ivar][irun],flagMID[ivar][irun],true);
// 	midvec->setValAndFlag(time[irun],midvarnevt[ivar],nevtinit[ivar][irun],flagMID[ivar][irun],true);
// 	midvec->setValAndFlag(time[irun],midvarrms[ivar],rmsMID[ivar][irun],flagMID[ivar][irun],true);
//       }
      
//     }
    
//     if( cMatacq[ivar]==0){
      
//       for (unsigned int irun=0;irun<nrunmax;irun++){
// 	if(cAPDPN[ivar]!=0){
// 	  midvec->setValAndFlag(time[irun],midvarcormean[ivar],valMID[ivar][irun],false,true);
// 	  midvec->setValAndFlag(time[irun],midvarcorrms[ivar],rmsMID[ivar][irun],false,true);
// 	}else{
// 	  midvec->setValAndFlag(time[irun],midvarcormean[ivar],valinit[ivar][irun],false,true);
// 	  midvec->setValAndFlag(time[irun],midvarcorrms[ivar],rmsinit[ivar][irun],false,true);
// 	}
// 	midvec->setValAndFlag(time[irun],midvarcornevt[ivar],nevtinit[ivar][irun],false,true);
//       }
      
//     }else{
      
//       assert(cMatacq[ivar]);
//       normMatacq[ivar]/=double(cMatacq[ivar]);

//       // get the intervals
//       MEIntervals* intervals_    = mtqIntervals( false , leaf_ );
     
      
//       // get the top interval
//       METimeInterval* topInterval = intervals_->topInterval();
      
//       double pente=-0.004;
//       vector< double > _corBeta;
//       _corBeta.push_back( 0.0 );
//       _corBeta.push_back( pente ); 
      
//       MECorrector2Var* cor_(0); 
//       cor_ = new MECorrector2Var( mtqVector_, midvec, topInterval, 
// 				  _type, corVar0, corVar1[ivar], 
// 				  MusEcal::iOneHundredPercent, MusEcal::iOneHundredPercent, 1,  
// 				  normMatacq[ivar], normAPDPN[ivar], _corBeta );
      
//       for (unsigned int irun=0;irun<nrunmax;irun++){
	
// 	// FIXME: check this carefully
// 	//==============================

// 	double correctionFactor = cor_->correctionFactor( valMatacq[irun] );
// 	if(normAPDPN[ivar]>0.0) correctionFactor=correctionFactor/normAPDPN[ivar];

// 	if(flagMID[ivar][irun] && flagMatacq[irun] && correctionFactor!=0 ) {
// 	  valCorMID[ivar].push_back(valMID[ivar][irun]/correctionFactor);
// 	  rmsCorMID[ivar].push_back(rmsMID[ivar][irun]/correctionFactor); 
// 	  flagCorMID[ivar].push_back(true);
// 	}else if(correctionFactor!=0){
// 	  valCorMID[ivar].push_back(valMID[ivar][irun]/correctionFactor);
// 	  rmsCorMID[ivar].push_back(rmsMID[ivar][irun]/correctionFactor); 
// 	  flagCorMID[ivar].push_back(false);
// 	}else{
// 	  valCorMID[ivar].push_back(valMID[ivar][irun]);
// 	  rmsCorMID[ivar].push_back(rmsMID[ivar][irun]); 
// 	  flagCorMID[ivar].push_back(false); 
// 	}
	
// 	midvec->setValAndFlag(time[irun],midvarcormean[ivar],valCorMID[ivar][irun],flagCorMID[ivar][irun],true);
// 	midvec->setValAndFlag(time[irun],midvarcornevt[ivar],nevtinit[ivar][irun],flagCorMID[ivar][irun],true);
// 	midvec->setValAndFlag(time[irun],midvarcorrms[ivar],rmsCorMID[ivar][irun],flagCorMID[ivar][irun],true);
	
//       }
//     }
//   }
//   return midvec;
// }

void MusEcal::fillNLSMaps(){


  while( !_nlsMap.empty() ) 
    { 
      delete _nlsMap.begin()->second;
      _nlsMap.erase( _nlsMap.begin() );
    }

  vector< MEChannel* > listOfChan_;
  listOfChan_.clear();
  curMgr()->tree()->getListOfDescendants( ME::iCrystal, listOfChan_ );

  unsigned int size_(0);

  if( _type==ME::iLaser )
    {
      size_=ME::iSizeNLS;
    }
  else{
    return;
  }
  
  cout << "Filling NLS Maps (number of C=" << listOfChan_.size() << ")" << endl;
  
  for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
    {
      MEChannel* leaf_ = listOfChan_[ichan];  
      
      if( _nlsMap.count(leaf_)==0 )
	{
	  _nlsMap[leaf_]=fillNLSVector(leaf_);
	}
    }
  cout << "...done." << endl;

  //
  // At higher levels
  
  cout << "Filling NLS Maps for other granularities" << endl;
  MERunManager* mgr_ = runMgr();
  
  for( MusEcal::RunIterator p=mgr_->begin(); p!=mgr_->end(); ++p )
    {
      MERun* run_ = p->second;
      ME::Time time = run_->time();
      
      for( int ig=ME::iSuperCrystal; ig>=ME::iLMRegion; ig-- )
	{
	  listOfChan_.clear();
	  if( ig==ME::iLMRegion ) listOfChan_.push_back( mgr_->tree() );
	  else  mgr_->tree()->getListOfDescendants( ig, listOfChan_ );
	  
	  for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
	    {
	      MEChannel* leaf_ = listOfChan_[ichan];
	      if( _nlsMap.count(leaf_)==0 )
		{
		  _nlsMap[leaf_] = new MEVarVector( size_ ); 
		}
	      MEVarVector* varVector_ = _nlsMap[leaf_];

	      varVector_->addTime( time );
	      for( unsigned ii=0; ii<=ME::iSizeNLS; ii++ )
		{
		  float val=0;
		  float n=0;
		  // loop on daughters
		  for( unsigned int idau=0; idau<leaf_->n(); idau++ )
		    {
		      float val_(0);
		      bool flag_=true;
		      
		      assert( _nlsMap[leaf_->d(idau)]
			      ->getValByTime( time, ii, val_, flag_ ) );
		      
		      if( val_>0 )
			{
			  n++;
			  val+=val_;
			}
		    }
		  if( n!=0 ) val/=n; 
		  varVector_->setVal( time, ii, val );
		}
	    }
	}
    }
  
  cout << "...done." << endl;

}

MEVarVector* MusEcal::nlsVector( MEChannel* leaf){

  if( _nlsMap.count( leaf ) !=0 ) return _nlsMap[leaf];
  else
    return 0;
}

MEVarVector* MusEcal::chooseNormalization( MEChannel* leaf_ ){

 if(_debug) cout<<" Entering fillNLSVector "<< leaf_->ig()<<endl;
  assert(leaf_->ig()==ME::iCrystal);
  
  std::vector<int> vars;
  METimeInterval* pninter = pnIntervals( vars, leaf_ );

  MEChannel* pnleaf_=leaf_;
  MEChannel* mtqleaf_=leaf_;

  if(leaf_->ig()>ME::iLMModule){
    while( pnleaf_->ig() != ME::iLMModule){
      pnleaf_=pnleaf_->m();
    }
  }
  if(leaf_->ig()>ME::iLMRegion){
    while( mtqleaf_->ig() != ME::iLMRegion){
      mtqleaf_=mtqleaf_->m();
    }
  }

  unsigned int size_(0);
  if( _type==ME::iLaser )
    {
      size_=ME::iSizeNLS;
    }

  MEVarVector* nlsvec= new MEVarVector( size_ );
    
  vector< float > val;
  vector< float > rms;
  vector< float > nevt;
  vector< bool > flag;
 
  vector< ME::Time > time; 

  vector<double> norm;
  vector<double> enorm;
  vector<bool> fnorm;
  vector<ME::Time> metimes;
  vector<unsigned int> times;
  
  vector<float> valcls;
  vector<float> rmscls;
  vector<bool> flagcls;
  vector<float> nevtcls;
  
  vector<float> valrab;
  vector<float> rmsrab;
  vector<float> flagrab;
  vector<float> normrab;
  vector<float> enormrab;
  
  vector<double> normbeg;
  vector<double> normend;
  vector<double> enormbeg;
  vector<double> enormend;
  vector<bool> flagbeg;
  vector<bool> flagend;
  
  double normTot=0.0;
  unsigned int cTot=0;
  
  MEVarVector* midvec = midVector(leaf_); 
  midvec->getTime( time );
  
  if(_debug) cout<<" TOTAL SIZE PN: "<<time.size()<<endl;
  METimeInterval* ki(0);
  unsigned int interval_(0);
  unsigned int ilevel=1;  
  

  for( ki=pninter->first(ilevel); ki!=0; ki=ki->next() )
    {

      // get norm to glue later on
      //==========================
      std::vector< double > norms;
      std::vector< double > enorms;
      std::vector< bool > flagnorms;

      if(interval_==0) metimes.push_back(ki->firstTime());
      metimes.push_back(ki->lastTime());
      if(_debug) cout <<"CHECK first "<<ki->firstTime()<< " "<<ki->lastTime()<< endl;
      if(_debug) cout <<"CHECK VAR "<<interval_<<" "<<vars[interval_]<< endl;

      // FIXME HARDCODED
      midvec->getNormsInInterval(vars[interval_],vars[interval_]+1,vars[interval_]+2,
				 _nbuf, _nave, ki, norms,enorms, flagnorms); 
      
      normbeg.push_back(norms[0]);
      normend.push_back(norms[1]); 
      enormbeg.push_back(enorms[0]);
      enormend.push_back(enorms[1]);
      flagbeg.push_back(flagnorms[0]);
      flagend.push_back(flagnorms[1]);

      // get val and fill with relevant normalisation pn
      //================================================
      vector< ME::Time > inttime;
      val.clear();
      flag.clear();
      rms.clear();
      nevt.clear();
      midvec->getTimeValAndFlag( vars[interval_]+2, inttime,  nevt, flag, ki); // +2= FIXME :)
      midvec->getTimeValAndFlag( vars[interval_]+1, inttime,  rms, flag, ki);// +1= FIXME :)
      midvec->getTimeValAndFlag( vars[interval_], inttime,  val, flag, ki);
      
      if(_debug) cout<<" INTERVAL SIZE PN: "<<interval_<<" "<<inttime.size()<<endl;
     
      int tmax=inttime.size()-1;
      if(ki->next()==0) tmax=inttime.size();
      
      for( int itime=0;itime<tmax;itime++){
	if(_debug2 && val[itime]==0.0 ) cout<<" bad values check1 "<< inttime[itime]<<" "<<val[itime]<<" "<<rms[itime]<<" "<<nevt[itime]<<" "<<flag[itime]<< endl;
	valcls.push_back(val[itime]);
	rmscls.push_back(rms[itime]);
	flagcls.push_back(flag[itime]);
	nevtcls.push_back(nevt[itime]);
      }
      interval_++;
    }
  assert(valcls.size() == time.size() );
  
  norm.push_back(1.0);
  enorm.push_back(0.0);
  fnorm.push_back(true);
  
  for(unsigned int j=1; j<normbeg.size();j++){

    double m2=normend[j-1];
    double m1=normbeg[j];
    double s2=enormend[j-1];
    double s1=enormbeg[j];
    double err=0.0;

    if( flagbeg[j] && flagend[j-1] && m1!=0){

      norm.push_back(m2/m1);
      err=sqrt( (s1*s1*m2*m2/(m1*m1)) + s2*s2 )/m1; 
      enorm.push_back(err);
      fnorm.push_back(true);

      // neglect here error on previous norm! 
   
    }else if(!flagbeg[j]){
      norm.push_back(norm[norm.size()-1]);
      enorm.push_back(enorm[norm.size()-1]);
      fnorm.push_back(false);

    }else if(!flagend[j-1]){
      
      // look for last good end interval
      bool found=false;
      int lastgood=j-2;
      for(int k=j-2;k>=0;k--){
	if(flagend[k]){
	  lastgood=k;
	  found=true;
	}
      }
      if(found && normbeg[j]){
	m2=normend[lastgood];
	s2=enormend[lastgood];
	err=sqrt( (s1*s1*m2*m2/(m1*m1)) + s2*s2 )/m1; 
	norm.push_back(m2/m1);
	enorm.push_back(err);
	fnorm.push_back(true);

      }else{
	
	norm.push_back(norm[norm.size()-1]);    
	enorm.push_back(enorm[norm.size()-1]);   
	fnorm.push_back(false);   

      }

    }else{
      
      norm.push_back(norm[norm.size()-1]);   
      enorm.push_back(enorm[norm.size()-1]); 
      fnorm.push_back(false);

    }  
  } 
  if(_debug){
    for(unsigned int j=0; j<norm.size();j++){ 
      cout<<"MYNORMS PN before prod: "<<j<<" "<< norm[j]<< endl;
    }
  }
  
  for(unsigned int j=1; j<norm.size();j++){
    
    if(fnorm[j]){
      norm[j]*=norm[j-1];
      enorm[j]*=norm[j-1]; 
    }else{
      norm[j]=norm[j-1];
      enorm[j]=enorm[j-1];
    }    
    // neglect error from previous norm here not to diverge
  }

  //  for(int j=0; j<normbeg.size();j++){
  //     if(_debug) 
  //       cout<<"NORMS PN BEG: "<<j<<" "<< normbeg[j]<<" "<<flagbeg[j]<< endl;
  //     if(_debug)
  //       cout<<"NORMS PN END: "<<j<<" "<< normend[j]<<" "<<flagend[j]<< endl;
  //   }

  if(_debug){
    for(unsigned int j=0; j<norm.size();j++){
      cout<<"MYNORMS PN after prod: "<<j<<" "<< norm[j]<< endl;
    }
  }
  
  // Now glue and normalize to PN intervals
  //========================================

  assert(metimes.size()==norm.size()+1);
  
  // loop on the varvec entries
  interval_=0;
  int firsttime=0;
  for(unsigned int itime=0;itime<time.size();itime++){
    for(unsigned int ime=firsttime;ime<metimes.size();ime++){
      if(metimes[ime]==time[itime]){
	times.push_back(itime);
	firsttime=ime+1;
	ime=metimes.size();
      }
    }
  }
  
  assert(metimes.size()==times.size());
  
  // Fill var vec entries:
  //=======================
  for(unsigned int ii=0;ii<times.size()-1;ii++){

    for(unsigned int jj=times[ii];jj<times[ii+1];jj++){

      valrab.push_back(valcls[jj]*norm[ii]);
      rmsrab.push_back(rmscls[jj]*norm[ii]); 
      normrab.push_back(norm[ii]);   
      enormrab.push_back(enorm[ii]); 
      
            
      // flag=2 for interval borders only
      if(flagcls[jj] && fnorm[ii] && jj==times[ii]) flagrab.push_back(2);   
      else if ( flagcls[jj] ) flagrab.push_back(1);
      else if ( !flagcls[jj] ) flagrab.push_back(0);

      if(_debug2 && valcls[jj]==0.0 ) cout<<" bad values check2 "<<time[jj] <<" "<< valcls[jj]<<" "<<flagcls[jj]<<" "<<flagrab[jj]<< endl;
     
      if(flagcls[jj]){
	normTot+=valrab[jj];
	cTot++;
      }
    }
  }

  // Fill last one:
  //===============
  
  unsigned int ii=times.size()-1;
  unsigned int jj=times[ii];
 
  if(_debug2 && valcls[jj]==0.0 ) cout<<" bad values check3 "<<time[jj]<<" "<< valcls[jj]<<" "<<flagcls[jj]<< endl;
  
  if(_debug) cout<< " CHECK LAST FILL PN "<< ii<<" "<<jj<<" "<<valcls[jj]<<" "<<norm[ii-1]<<" "<<enorm[ii-1]<<endl;
  valrab.push_back(valcls[jj]*norm[ii-1]);
  rmsrab.push_back(rmscls[jj]*norm[ii-1]);    
  if( flagcls[jj] ) flagrab.push_back(1);
  else flagrab.push_back(0);
  normrab.push_back(norm[ii-1]);
  enormrab.push_back(enorm[ii-1]);
  if(flagcls[jj]){
    normTot+=valrab[jj];
    cTot++;
  }
  
  
  
  if(cTot>=1) normTot/=double(cTot);
  else normTot=1.0;
  normTot=1.0;


  assert(flagcls.size()==nevtcls.size());
  assert(valcls.size()==nevtcls.size());
  assert(rmscls.size()==nevtcls.size());

  assert(normrab.size()==nevtcls.size());
  assert(enormrab.size()==nevtcls.size());
  assert(valrab.size()==nevtcls.size());
  assert(flagrab.size()==nevtcls.size());

  for(unsigned int ii=0;ii<time.size();ii++){
    nlsvec->setValAndFlag(time[ii],ME::iCLS_MEAN,valcls[ii],flagcls[ii],true);
    nlsvec->setValAndFlag(time[ii],ME::iCLS_NEVT,nevtcls[ii],flagcls[ii],true);
    nlsvec->setValAndFlag(time[ii],ME::iCLS_RMS,rmscls[ii],flagcls[ii],true);
    nlsvec->setValAndFlag(time[ii],ME::iCLS_NORM,normrab[ii],flagcls[ii],true);
    nlsvec->setValAndFlag(time[ii],ME::iCLS_ENORM,enormrab[ii],flagcls[ii],true);
    nlsvec->setValAndFlag(time[ii],ME::iCLS_NMEAN,valrab[ii],flagcls[ii],true);
    nlsvec->setValAndFlag(time[ii],ME::iCLS_FLAG,flagrab[ii],flagcls[ii],true);
 
  }
  
  return nlsvec;
}

MEVarVector* MusEcal::fillNLSVector( MEChannel* leaf_ ){

  // get the intervals
  MEIntervals* intervals_    = mtqIntervals( false , leaf_ );
  METimeInterval* topInterval = intervals_->topInterval();

  MEVarVector*nlsvec=fillNLSVector(leaf_, topInterval);
  return nlsvec;
}

MEVarVector* MusEcal::fillNLSVector( MEChannel* leaf_ , 
				     METimeInterval *intervals){

  MEVarVector* nlsvec = chooseNormalization(leaf_);

  // get the intervals

  vector<int> ivarin;
  vector<int> ivarout;

  ivarin.push_back(ME::iCLS_MEAN);
  ivarin.push_back(ME::iCLS_RMS);
  ivarin.push_back(ME::iCLS_NEVT);
  ivarin.push_back(ME::iCLS_NORM);
  ivarin.push_back(ME::iCLS_NMEAN);
  ivarin.push_back(ME::iCLS_ENORM);
  ivarin.push_back(ME::iCLS_FLAG);

  ivarout.push_back(ME::iNLS_MEAN);
  ivarout.push_back(ME::iNLS_RMS);
  ivarout.push_back(ME::iNLS_NEVT);
  ivarout.push_back(ME::iNLS_NORM);
  ivarout.push_back(ME::iNLS_NMEAN);
  ivarout.push_back(ME::iNLS_ENORM);
  ivarout.push_back(ME::iNLS_FLAG);
		    
  rabouteNLS( intervals, nlsvec, nlsvec, ivarin, ivarout, _nbuf, _nave, _nbad);

  return nlsvec;
}


void 
MusEcal::rabouteNLS( MEIntervals* intervals, 
		     MEVarVector* varvecin, MEVarVector* varvecout,
		     vector<int> ivarin, vector<int> ivarout,
		     unsigned int nbuf, unsigned int nave, unsigned int nbad)
{ 

  METimeInterval* topInterval = intervals->topInterval();
  rabouteNLS(topInterval, varvecin, varvecout, ivarin, ivarout, nbuf, nave, nbad );
  
}

void 
MusEcal::rabouteNLS( METimeInterval* intervals, 
		     MEVarVector* varvecin, MEVarVector* varvecout,
		     vector<int> ivarin, vector<int> ivarout,
		     unsigned int nbuf, unsigned int nave, unsigned int nbad)
{ 
  assert(ivarin.size()==ivarout.size());
  assert(ivarin.size()==ME::iSizeNLS/2);

  vector< ME::Time > time;
  
  vector<float> valcls;
  vector<float> nevtcls;
  vector<float> rmscls;  
  vector<float> normcls; 
  vector<float> enormcls;
  vector<float> valnormcls;
  vector<float> myflagcls;
  vector<bool>  flagcls;
  vector<bool>  flag;
  vector<bool>  flagclsrab;

  vector<float> normrab;
  vector<float> enormrab;
  vector<float> valnormrab;
  vector<float> flagrab;
  vector<float> flagrabgood;

  vector<float> valrabgood;
  vector<float> rmsrabgood;
  vector<float> normrabgood;
  vector<float> enormrabgood;
  vector<float> valnormrabgood;


  vector<double> normbeg;
  vector<double> normend;
  vector<double> enormbeg;
  vector<double> enormend;
  vector<bool> flagbeg;
  vector<bool> flagend;
  
  double normTot=0.0;
  unsigned int cTot=0;

  vector<double> norm;
  vector<double> enorm;
  vector<bool> fnorm;
  vector<ME::Time> metimes;
  vector<unsigned int> times;

  // FIXME HARCODED
  varvecin->getTime( time, intervals );
  varvecin->getValAndFlag(ivarin[0], time, valcls, flagcls);
  varvecin->getValAndFlag(ivarin[1], time, rmscls, flag);
  varvecin->getValAndFlag(ivarin[2], time, nevtcls, flag);
  varvecin->getValAndFlag(ivarin[3], time, normcls, flagclsrab);
  varvecin->getValAndFlag(ivarin[4], time, valnormcls, flagclsrab);
  varvecin->getValAndFlag(ivarin[5], time, enormcls, flagclsrab);
  varvecin->getValAndFlag(ivarin[6], time, myflagcls, flag);


  METimeInterval* ki(0);
  unsigned int interval_(0);
  unsigned int ilevel=1;

  if(_debug) cout<<" TOTAL SIZE: "<<time.size()<<endl;

  for( ki=intervals->first(ilevel); ki!=0; ki=ki->next() )
    {
      std::vector< double > norms;
      std::vector< double > enorms;
      std::vector< bool > flagnorms;
      if(interval_==0) metimes.push_back(ki->firstTime());
      metimes.push_back(ki->lastTime());

      // FIXME HARDCODED!!!!!!!!

      varvecin->getNormsInInterval(ivarin[4],ivarin[1],ivarin[2],
				   nbuf, nave, ki, norms,enorms,flagnorms);

      normbeg.push_back(norms[0]);
      normend.push_back(norms[1]);
      enormbeg.push_back(enorms[0]);
      enormend.push_back(enorms[1]);
      flagbeg.push_back(flagnorms[0]);
      flagend.push_back(flagnorms[1]);
      interval_++;

    }

  // first interval normalized to 1 for now

  norm.push_back(1.0);
  enorm.push_back(0.0);
  fnorm.push_back(true);

  for(unsigned int j=1; j<normbeg.size();j++){
    
    if(_debug) cout<< " NORMS "<<j-1<<" BEGj:" <<normbeg[j]<<" "<<flagbeg[j]<<" ENDj-1:"<<normend[j-1]<<" "<<flagend[j-1] <<endl;
    
    double m2=normend[j-1];
    double m1=normbeg[j];
    double s2=enormend[j-1];
    double s1=enormbeg[j];
    double err=0.0;

    if( flagbeg[j] && flagend[j-1] && m1!=0){
      norm.push_back(m2/m1);
      err=sqrt( (s1*s1*m2*m2/(m1*m1)) + s2*s2 )/m1; 
      enorm.push_back(err);
      fnorm.push_back(true);
    }
    else if(!flagbeg[j]){
      norm.push_back(norm[norm.size()-1]);
      enorm.push_back(enorm[norm.size()-1]);
      fnorm.push_back(false);
    }
    else if(!flagend[j-1]){
      
      // look for last good end interval
      bool found=false;
      int lastgood=j-2;
      for(int k=j-2;k>=0;k--){
	if(flagend[k]){
	  lastgood=k;
	  found=true;
	}
      }
      if(found && normbeg[j]){
	m2=normend[lastgood];
	s2=enormend[lastgood];
	err=sqrt( (s1*s1*m2*m2/(m1*m1)) + s2*s2 )/m1; 
	norm.push_back(m2/m1);
	enorm.push_back(err);
	fnorm.push_back(true);
      }else{
	norm.push_back(norm[norm.size()-1]);   
	enorm.push_back(enorm[norm.size()-1]);   
	fnorm.push_back(false);
      }
    }else{
      norm.push_back(norm[norm.size()-1]);
      enorm.push_back(enorm[norm.size()-1]);
      fnorm.push_back(false);
    }  
  }
  if(_debug) {
    for(unsigned int j=0; j<norm.size();j++){
      cout<<"RAB MYNORMS before prod: "<<j<<" "<< norm[j]<<" "<<fnorm[j]<< endl;
    }
  }
  for(unsigned int j=1; j<norm.size();j++){
    if(fnorm[j]){
      norm[j]*=norm[j-1];
      enorm[j]*=norm[j-1];
    }else{
      norm[j]=norm[j-1];
      enorm[j]=enorm[j-1];
    } 
    // neglect error from previous norm here not to diverge
  }
  
  if(_debug) {
    for(unsigned int j=0; j<norm.size();j++){
      cout<<"RAB MYNORMS after prod: "<<j<<" "<< norm[j]<<" "<< fnorm[j]<<endl;
    }
  }

  assert(norm.size()==interval_);
  assert(metimes.size()==norm.size()+1);
  
  // loop on the varvec entries
  interval_=0;
  int firsttime=0;
  for(unsigned int itime=0;itime<time.size();itime++){
    for(unsigned int ime=firsttime;ime<metimes.size();ime++){
      if(metimes[ime]==time[itime]){ 
	
	times.push_back(itime);
	firsttime=ime+1;
	ime=metimes.size();
      }
    }
  }
  
  assert(metimes.size()==times.size());
  
  
  // loop on the varvec entries
  for(unsigned int ii=0;ii<times.size()-1;ii++){

    for(unsigned int jj=times[ii];jj<times[ii+1];jj++){
      
      valnormrab.push_back(valcls[jj]*normcls[jj]*norm[ii]);    
      normrab.push_back(normcls[jj]*norm[ii]); 
      enormrab.push_back( sqrt( (normcls[jj]*enorm[ii])*(normcls[jj]*enorm[ii]) + (enormcls[jj]*norm[ii])*(enormcls[jj]*norm[ii]) ) );
      
      // flag=2 for interval borders only
      if(jj==times[ii] && _debug)
	cout<<"new Interval: "<<flagcls[jj]<<" "<<fnorm[ii]<<" "<< myflagcls[jj]<<" "<<times[ii]<<endl;
      
      if( flagcls[jj] && fnorm[ii] && jj==times[ii]) flagrab.push_back(2.0);  
      else flagrab.push_back(myflagcls[jj]);

      // else if ( flagcls[jj] ) flagrab.push_back(1.0);
      // else if ( !flagcls[jj] ) flagrab.push_back(0.0);
      
      if(_debug2 && valcls[jj]==0.0 ) cout<<" bad values check4 "<<time[jj]<<" "<<valcls[jj]<<" "<<valnormrab[jj]<<" "<< myflagcls[jj]<<" "<<flagrab[jj]<< endl;

      if(flagcls[jj]){
	normTot+=valcls[jj];
	cTot++;
      }
    }
  }
  
  // Fill last one:
  //===============
  
  unsigned int ii=times.size()-1;
  unsigned int jj=times[ii];   
  normrab.push_back(normcls[jj]*norm[ii-1]);
  enormrab.push_back( sqrt( (normcls[jj]*enorm[ii-1])*(normcls[jj]*enorm[ii-1]) + (enormcls[jj]*norm[ii-1])*(enormcls[jj]*norm[ii-1])));
  
  if(_debug) cout<< " CHECK LAST FILL "<< ii<<" "<<jj<<" "<<valcls[jj]<<" "<<normcls[jj]<<" "<<norm[ii-1]<<" "<<enorm[ii-1]<<endl;
  valnormrab.push_back(valcls[jj]*normcls[jj]*norm[ii-1]);
  if(flagcls[jj]) flagrab.push_back(myflagcls[jj]);
  else flagrab.push_back(0);
  
  if(_debug2 && valcls[jj]==0.0) cout<<" bad values check5 "<<time[jj]<<" "<<valcls[jj]<<" "<<valnormrab[jj]<<" "<< myflagcls[jj]<<" "<<flagrab[jj]<< endl;
  
  if(flagcls[jj]){
    normTot+=valcls[jj];
    cTot++;
  }
  

  if(cTot>=1) normTot/=double(cTot);
  else normTot=1.0;

//   double lastGoodVal=1.0;
//   double lastGoodNormVal=1.0;
//   double lastGoodRMS=0.0;
//   double lastGoodNevt=0.0;
//   double lastGoodNorm=1.0;
//   double lastGoodENorm=0.0;
//   bool   doesLastGoodExist=false;
 

  for(unsigned int ii=0;ii<time.size();ii++){

    if(valcls[ii]<0.2) flagcls[ii]=false; // FIXME
    // this should not be needed indeed...
    
      
       valrabgood.push_back(valcls[ii]);
       rmsrabgood.push_back(rmscls[ii]);
       normrabgood.push_back(normrab[ii]);
       enormrabgood.push_back(enormrab[ii]);
       valnormrabgood.push_back(valcls[ii]*normrab[ii]);


  //   if(flagcls[ii]){
      
//       valrabgood.push_back(valcls[ii]);
//       rmsrabgood.push_back(rmscls[ii]);
//       normrabgood.push_back(normrab[ii]);
//       enormrabgood.push_back(enormrab[ii]);
//       valnormrabgood.push_back(valcls[ii]*normrab[ii]);
      
//       lastGoodVal=valcls[ii];
//       lastGoodRMS=rmscls[ii];
//       lastGoodNevt=nevtcls[ii];
//       lastGoodNorm=normrab[ii];
//       lastGoodENorm=enormrab[ii];
//       lastGoodNormVal=valcls[ii]*normrab[ii];
//       doesLastGoodExist=true;
      
//     }else if( doesLastGoodExist ){
      
//       valrabgood.push_back(lastGoodVal);
//       rmsrabgood.push_back(lastGoodRMS);
//       normrabgood.push_back(lastGoodNorm);
//       enormrabgood.push_back(lastGoodENorm);
//       valnormrabgood.push_back(lastGoodNormVal);
      
//     }else(if(ii>0){
      
//       valrabgood.push_back(valcls[ii-1]);
//       rmsrabgood.push_back(rmscls[ii-1]);
//       normrabgood.push_back(normrab[ii-1]);
//       enormrabgood.push_back(enormrab[ii-1]);
//       valnormrabgood.push_back(valcls[ii-1]*normrab[ii-1]);
      
//     }
    

    varvecout->setValAndFlag(time[ii],ivarout[0],valcls[ii],flagcls[ii],true);
    varvecout->setValAndFlag(time[ii],ivarout[1],rmscls[ii],flagcls[ii],true);
    varvecout->setValAndFlag(time[ii],ivarout[2],nevtcls[ii],flagcls[ii],true);
    varvecout->setValAndFlag(time[ii],ivarout[3],normrab[ii],flagcls[ii],true); // Norm
    varvecout->setValAndFlag(time[ii],ivarout[4],valnormrab[ii],flagcls[ii],true);
    varvecout->setValAndFlag(time[ii],ivarout[5],enormrab[ii],flagcls[ii],true); // Norm
    varvecout->setValAndFlag(time[ii],ivarout[6],flagrab[ii],flagcls[ii],true); // Flag
  }

  if(_debug) cout <<" CHECK SIZES RAB GOOD: "<<ii<<" "<<valrabgood.size()<<" "<<rmsrabgood.size()<<" "<<normrabgood.size()<<" "<< valnormrabgood.size()<<endl;
  
  assert(valcls.size()==flagcls.size());
  assert(normrab.size()==flagrab.size());
  assert(valnormrab.size()==flagrab.size());
  assert(enormrab.size()==flagrab.size());
  assert(flagcls.size()==flagrab.size());
  
}


void
MusEcal::writeGlobalHistograms()
{
  map< TString, TH1* >::iterator it;
  for( it=_eb_m.begin(); it!=_eb_m.end(); it++ )
    {
      it->second->Write();
    }
  for( it=_ee_m.begin(); it!=_ee_m.end(); it++ )
    {
      it->second->Write();
    }
}

TString
MusEcal::mgrName( int lmr, int type, int color )
{
  TString out_;
  out_ += type; 
  out_ += "_"; out_ += color; 
  out_ += "_"; 
  if( lmr<10 ) out_+="0";
  out_ += lmr; 
  return out_;
}

void MusEcal::corMapFwhmMID(vector<double>& slope, vector<double>& chi2){

  slope.clear();
  chi2.clear();
  vector< MEChannel* > listOfChan_;
  
  listOfChan_.clear();
  curMgr()->tree()->getListOfDescendants( ME::iCrystal, listOfChan_ );

  cout << "Getting Fwhm Vs MID correlation (number of C=" << listOfChan_.size() << ")" << endl;
  
  for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
    {
      
      MEChannel* leaf_ = listOfChan_[ichan];  
      slope.push_back(correlationFwhmMID(leaf_).first);
      chi2.push_back(correlationFwhmMID(leaf_).second);
    }
}



pair <double,double> MusEcal::correlationFwhmMID(MEChannel* leaf_){
  
  assert(leaf_->ig()==ME::iCrystal);
  
  MEChannel* mtqleaf_=leaf_;
  
  if(leaf_->ig()>ME::iLMRegion){
    while( mtqleaf_->ig() != ME::iLMRegion){
      mtqleaf_=mtqleaf_->m();
    }
  }
  
  MEVarVector* mtqVector_ = curMgr()->mtqVector(mtqleaf_); 
  MEVarVector* midVector_ = midVector(leaf_); 
  
  int Var0 = ME::iMTQ_FWHM; 
  int Var1 = ME::iMID_MEAN;

  int fitDeg=1;
  int zoomRangeFwhm=iTenPercent;
  int zoomRangeMID=iPercent;
  
  vector<double> res=correlation2Var(mtqVector_,midVector_,Var0,Var1,zoomRangeFwhm,zoomRangeMID,fitDeg);
  
  if(res.size()==1) return pair<double,double> (0.0,0.0);
  
  double slope=res[res.size()-1];
  double chi2=res[1];
  return pair<double,double> (slope,chi2);
}

vector<double>
MusEcal::correlation2Var( MEVarVector* vec0,  MEVarVector* vec1, int var0, int var1, int zoom0, int zoom1, int fitDegree)
{


  vector< float > val[2];
  vector< bool > flag[2];
  vector< float > val0;
  vector< bool  > flag0;
  vector< float > val1;
  vector< bool  > flag1;
  vector< float > nevt;
  vector< ME::Time > time;
  double norm0=1.0;
  double norm1=1.0;
  double norm[2];
 
  vec1->getTime( time );
  vec0->getValFlagAndNorm(var0, time, val0, flag0, norm0);
  vec1->getValFlagAndNorm(var1, time, val1, flag1, norm1);
  assert(val0.size()==val1.size());

  val[0]=val0;
  val[1]=val1;
  flag[0]=flag0;
  flag[1]=flag1;
  norm[0]=norm0;
  norm[1]=norm1;
 
  int varZoom[ 2 ]  = { zoom0, zoom1 };
  
  unsigned int nrun=val0.size();
  vector<bool> keep(nrun,true);
  
  double y_[2][nrun];

  for( unsigned var=0; var<2; var++ )
    {
      double zr = zoomRange[ varZoom[var] ];
      
      for( unsigned ii=0; ii<nrun; ii++ )
	{
	  bool   oo = flag[var][ii];
	  double yy = (val[var][ii]-norm[var])/norm[var]/zr;
	  if( !oo ) keep[ii] = false;
	  if( yy<-1 || yy>1 ) keep[ii] = false;
	  y_[var][ii] = yy;
	}
    }
  double minx = -1;
  double maxx = +1;
  
  vector<double> xgood;
  vector<double> ygood;
  
  for( unsigned ii=0; ii<nrun; ii++ ){
    if( keep[ii] )
      {
	xgood.push_back( y_[0][ii] );
	ygood.push_back( y_[1][ii] );
      }
  }
  vector<double> results;
  results.clear();
  results.push_back(double(fitDegree));

  unsigned NN = xgood.size();
  if(NN==0) return results;
  double XX[NN]; double YY[NN]; 
  for( unsigned jj=0; jj<NN; jj++ ) { XX[jj]=xgood[jj]; YY[jj]=ygood[jj]; }
  TGraph* gr = new TGraph( NN, XX, YY );
  TString dumstr = "pol"; dumstr += fitDegree;
  TF1* f1 = new TF1("f1",dumstr.Data(),minx,maxx);
  gr->Fit("f1","QN0");
  double chi2=f1->GetChisquare();
  double realSlope=0.0;
  results.push_back(chi2);
  for(int i=0;i<fitDegree+1;i++){ 
    results.push_back(f1->GetParameter(i));
  }

  if(fitDegree<=2){
    float slope=f1->GetParameter(1);
    float ratio;
    if(zoomRange[varZoom[0]]*norm[0]!=0){
      ratio= zoomRange[varZoom[1]]*norm[1]/(zoomRange[varZoom[0]]*norm[0]);
      realSlope=ratio*slope;
      //cout << "Real Slope = "<<realSlope<< endl;
      if(fitDegree==2) cout << "Real 2nd order = "<<ratio*ratio*f1->GetParameter(2)<< endl;
    }
  }
  results.push_back(realSlope);
  delete f1;
  delete gr;
  return results;

}


TH2*
MusEcal::getEBExtraHistogram(int ivar)
{

  TH2* h2_=0;
  
  if(ivar > ME::iSizeMID){
    cout<<" getEBExtraHistograms: variable is not valid. has to be less than "<<ME::iSizeMID<< endl;
    return h2_;
  }
  if(  _type!=ME::iLaser ){
    cout<< " getEBExtraHistograms: type has to be laser"<< endl;
    return h2_;
  }
  cout << " Filling Extra Histogram for Variable:" <<ME::MIDPrimVar[ivar]<< endl;
  
  _febgeom = TFile::Open( ME::path()+"geom/ebgeom.root" );
  assert( _febgeom!=0 );
  _eb_h     = (TH2*) _febgeom->Get("eb");
  _eb_h->SetStats( kFALSE );
  _eb_h->GetXaxis()->SetTitle("ieta");
  _eb_h->GetXaxis()->CenterTitle();
  _eb_h->GetYaxis()->SetTitle("iphi");
  _eb_h->GetYaxis()->CenterTitle();
  
  TString varName_;
  varName_=ME::MIDPrimVar[ivar];

  TString str_="EXT-"+varName_;
  h2_ = (TH2*)_eb_h->Clone(str_);	  
  
  // fill 2D histogram

  TString rundir_;
  MERunManager* firstMgr_ = _runMgr.begin()->second;
  if( firstMgr_!=0 )
    {
      MERun* firstRun_ = firstMgr_->curRun();
      if( firstRun_!=0 ) 
	{
	  rundir_ += firstRun_->rundir();
	}
    }
  TString titleW;

  titleW = ME::type[_type];
  if( _type==ME::iLaser )
    {
      titleW+=" "; titleW+=ME::color[_color];
    }
  titleW+=" ECAL Barrel";
  titleW+=" XXXXX";
  titleW+=" "; titleW+=rundir_;


  vector< MEChannel* > vec;
  for( int ism=1; ism<=36; ism++ )
    {
      int idcc=MEEBGeom::dccFromSm( ism );
      int ilmr;
      MERunManager* mgr_;
      MERun* run_;
      for( int side=0; side<2; side++ )
	{
	  ilmr=ME::lmr( idcc, side );
	  setLMRegion( ilmr );
	  mgr_=runMgr( ilmr );
	  if( mgr_==0 ) continue;
	  run_= mgr_->curRun();
	  if( run_==0 ) continue;

	  //cout <<"here ok ism " <<ism<<" side "<<side<<endl;
	  vec.clear();
	  mgr_->tree()->getListOfChannels( vec );
	  ME::Time curtime= run_->time();
	  MEVarVector* midvec;
	  for( unsigned int ii=0; ii<vec.size(); ii++ )
	    {
	      MEChannel* leaf_ = vec[ii];
	      int ieta = leaf_->ix();
	      int iphi = leaf_->iy();
	      //MEVarVector* midvec=fillMIDVector(leaf_);
	      midvec = midVector(leaf_);
	      float val;
	      bool flag;
	      midvec->getValByTime(curtime, ivar , val, flag);
	      
	      TString title_ = titleW;
	      title_.ReplaceAll("XXXXX",str_);
	      h2_->SetTitle(title_);
	      h2_->Fill( ieta, iphi, val ); 
	      
	    }
	}
    }
  cout << " Done." << endl;
  return h2_;
}
