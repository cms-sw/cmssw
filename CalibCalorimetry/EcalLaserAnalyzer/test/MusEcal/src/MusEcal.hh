#ifndef MusEcal_hh
#define MusEcal_hh

#include <map>

//
// MusEcal : Monitoring and Useful Survey of CMS Ecal
//                  Clever Analysis of Laser
//
// Authors  : Gautier Hamel de Monchenault and Julie Malcles, Saclay 
//
#include <TROOT.h>
#include <TH1.h>
#include <TH2.h>
#include <TTree.h>
#include <TFile.h>

#include "../../interface/ME.h"

class MERunManager;
class MERun;
class MEChannel;
class METimeInterval;
class MEIntervals;
class MECorrector2Var;
class MEVarVector;
class TCalibData;

#define NBUF 3;
#define NBAD 2;
#define NAVE 15;

class MusEcal
{

public:

  typedef std::map< ME::Time, MERun* >              RunMap;
  typedef std::map< ME::Time, float >               FloatTimeMap;
  typedef std::map< ME::Time, bool >                BoolTimeMap;
  typedef std::pair< float, bool >              Var;
  typedef std::map< ME::Time, Var >                 VarTimeMap;
  typedef std::vector<Var>                      VarVec;
  typedef std::map< ME::Time, VarVec* >             VarVecTimeMap;
  typedef RunMap::iterator             RunIterator;
  typedef RunMap::const_iterator  RunConstIterator;

  enum HistCateg  { iH_APD=0, iH_PN, iH_MTQ, iSizeHC };
  enum HistType   { iHIST=0, iVS_CHANNEL, iMAP, iSizeHT }; 

  // MusEcal: Laser variables 
  // CLS = chosen laser signal
  // NLS = final stuff

  enum {  iCLS, iCLSN, iNLS, iNLSN, iCLSNORM, iNLSNORM,
	  iMIDA, iMIDB,  iMID, 
	  iAPDoPNA, iAPDoPNB, iAPDoPN, 
	  iAPDoPNACOR, iAPDoPNBCOR, iAPDoPNCOR, 
	  iAPDABFIXoPNACOR, iAPDABFIXoPNBCOR, iAPDABFIXoPNCOR, 
	  iAPDABFIToPNACOR, iAPDABFIToPNBCOR, iAPDABFIToPNCOR, 
	  iAPDoPNANevt, iAPDoPNBNevt,
	  iAPDoAPDA, iAPDoAPDB, iAPDoAPDANevt, iAPDoAPDBNevt,  
	  iAPD, iAPDTime, iAPDNevt, iAPDTimeNevt,   
	  iPNA, iPNB, iPNANevt, iPNBNevt,iPNARMS, iPNBRMS, iPNBoPNA, iShapeCorPNA,iShapeCorPNB,
	  iAlphaBeta, iShapeCorAPD, iShapeCorRatio,
	  iMTQTrise, iMTQAmpl, iMTQFwhm, iMTQFw10, iMTQFw05, iMTQTime,
	  iSizeLV };  
  
  // MusEcal: TP variables
  enum { iTPAPD_0, iTPAPD_1, iTPAPD_2, iTPPNA_0, iTPPNA_1, iTPPNB_0, iTPPNB_1, iSizeTPV };  

  // Zoom
  enum { iOneHundredPercent, iFiftyPercent, iThirtyPercent, iTenPercent,
	 iFivePercent, iThreePercent, iPercent, 
	 iFivePerMil, iThreePerMil, iPerMil, iZero };


  static TString historyVarTitle[iSizeLV];
  static TString historyVarName[iSizeLV];
  static int iGVar[iSizeLV];
  static int historyVarZoom[ME::iSizeC][iSizeLV];
  static int historyVarColor[iSizeLV];
  static TString historyTPVarName[iSizeTPV];
  static TString historyTPVarTitle[iSizeTPV];
  static int iGTPVar[iSizeTPV];
  static int historyTPVarColor[iSizeTPV];
  static int historyTPVarZoom[iSizeTPV];
  static TString zoomName[ iZero ];
  static double zoomRange[ iZero ];
  static TString mgrName( int lmr, int type, int color );
  static int  firstRun;
  static int  lastRun;
  static bool fillNLS;
  static int  LMR;
  //  static int  corVar;

  // contructors/destructor
  MusEcal( int type=ME::iLaser, int color=ME::iBlue );
  virtual  ~MusEcal();

  // access to run manager for given laser monitoring region
  MERunManager* runMgr( int lmr ) { return runMgr(  lmr, _type, _color ); }
  MERunManager* runMgr()          { return runMgr( _lmr, _type, _color ); }

  // access to run manager for current laser monitoring region
  MERunManager* curMgr()          { return runMgr( _lmr ); }

  // set type (Laser, TestPulse, Pedestal) and color if applicable
  virtual void setType( int type, int color=ME::iBlue );

  // set laser monitoring region
  virtual void setLMRegion( int lmr );

  // set current time
  virtual void setTime( ME::Time );
  
  // set granularity and channel
  virtual void setChannel( MEChannel* );
  virtual void setChannel( int ig, int ieta, int iphi, bool useEtaPhi=true );

  // channel navigation
  void oneLevelUp();

  // set default time
  void setDefaultTime();

  // set run & sequence in run
  void setRunAndSequence( unsigned int run, int seq=1 );

  // next sequence
  bool nextSequence();

  // set laser monitoring variable
  void setVar( int var=MusEcal::iAPD ) { _var = var; _zoom = 0; }
  
  // dump vector for current channel and variable (ascii)
  void dumpVector( int ivar );

  // histograms
  void histConfig();
  void bookHistograms();
  void bookEBAPDHistograms();
  void bookEBPNHistograms();
  void bookEEAPDHistograms();
  void bookEEPNHistograms();

  TH2* getEBExtraHistogram(int ivar);

  void fillHistograms();
  void fillEBGlobalHistograms();
  void fillEBLocalHistograms();
  void fillEEGlobalHistograms();
  void fillEELocalHistograms();
  void writeGlobalHistograms();

  void corMapFwhmMID(std::vector<double>& slope, std::vector<double>& chi2);

  METimeInterval* pnIntervals( std::vector<int>& iapdopn , MEChannel* leaf);
  MEIntervals* mtqIntervals( bool createcor , MEChannel* leaf);
  MEIntervals* tpIntervals( MEChannel* leaf );
  std::vector < std::pair<ME::Time, ME::Time> > tpFlagIntervals( MEChannel *leaf, int pnNum, double cutStep, double cutSlope );
  MEIntervals* intervals( MEChannel* leaf );

  MEVarVector* chooseNormalization( MEChannel* leaf_ );
  MEVarVector* fillNLSVector( MEChannel* ,  METimeInterval *intervals ); //JM 
  MEVarVector* midVector( MEChannel* ); //JM
  
  
  // verbosity
  static bool verbose;

protected:

  virtual void refresh();

  // map of run managers
  std::map< TString, MERunManager* > _runMgr;

  // access to run managers
  MERunManager* runMgr( int lmr, int type, int color );

  // current monitoring region
  int _lmr;
  bool isBarrel();

  // current run type (Laser, Test Pulse, Pedestal)
  int _type; 

  // current color
  int _color;

  // current channel or group of channels
  MEChannel* _leaf; 

  // current level of detector  
  int _ig;         

  // reference time
  ME::Time _time;

  // current laser monitoring variable
  int _var;
  
  // current zoom
  int _zoom;

  // debug
  bool _debug;
  bool _debug2;
  unsigned _nbuf ;
  unsigned _nbad ;
  unsigned _nave ;

  // GUI
  bool _isGUI;

  //
  // Histograms
  //
  bool _histoBooked;
  bool _ebHistoBooked;
  bool _eeHistoBooked;

  TFile* _febgeom;
  TH2 *_eb_h, *_eb_loc_h;
  std::map< TString, TH1* > _eb_m;
  std::map< TString, TH1* > _eb_loc_m;
  std::map< TString, TH1* > _eb_xm;
  std::map< TString, TH1* > _eb_loc_xm;

  TFile* _feegeom;
  TH2* _ee_h; 
  TH2* _ee_loc_h[10];
  std::map< TString, TH1* > _ee_m;
  std::map< TString, TH1* > _ee_loc_m;

  // histogam limits
  std::map< TString,  int  > _eb_nbin;
  std::map< TString, float > _eb_min;
  std::map< TString, float > _eb_max;
  std::map< TString,  int  > _ee_nbin;
  std::map< TString, float > _ee_min;
  std::map< TString, float > _ee_max;
  int   hist_nbin( TString& );
  float hist_min(  TString& );
  float hist_max(  TString& );

  TTree* _seq_t;    // sequence rootuple
  Int_t _seq_run;   // run number
  Int_t _seq_lb;    // lumi block of first LMR
  Int_t _seq_tbeg;  // time at beginning of sequence
  Int_t _seq_tlmr[92]; // array of times, -1 LMR not present

  TBranch        *b_seq_run;   //!
  TBranch        *b_seq_lb;   //!
  TBranch        *b_seq_tbeg;   //!
  TBranch        *b_seq_tlmr[92];   //!

  // PN linearity corrector 
  TCalibData *_calibData;

  // Set variables for validity intervals

  bool _mtqTwoVar;
  int  _mtqLevel;
  int  _mtqVar0;
  int  _mtqVar1;

  bool _tpTwoVar;
  int  _tpLevel;
  int  _tpVar0;
  int  _tpVar1;

  bool _pnTwoVar;
  int  _pnLevel;
  int  _pnVar0;
  int  _pnVar1;

  void setMtqVar( int mtqVar0, int mtqVar1, int mtqLevel );
  void setTPVar( int tpVar0, int tpVar1, int tpLevel );
  void setPNVar(  int  pnVar0, int  pnVar1, int  pnLevel );

  
  // Compute validity intervals
  
  std::map<MEChannel*, MEIntervals*> _intervals[ME::iSizeC][ME::iSizeT];
  std::map<MEChannel*, METimeInterval*> _pnintervals[ME::iSizeC];


  // Choice of variables in PN intervals

  std::map<MEChannel*, std::vector<int> > _pnvarmap[ME::iSizeC];


  virtual void buildMtqIntervals( bool createcor , MEChannel *leaf );
  virtual void buildTPIntervals(MEChannel *leaf);
  virtual void buildPNIntervals( std::vector<int>& iapdopn, MEChannel *leaf);

  // Define intermediate and NLS maps 
  
  std::map< MEChannel*, MEVarVector* >  _midMap;
  std::map< MEChannel*, MEVarVector* >  _nlsMap;

  
  // Fill intermediate and NLS maps (should be private)

  void fillMIDMaps();
  MEVarVector* fillMIDVector( MEChannel*  ); //JM
 
  void fillNLSMaps( );
  MEVarVector* nlsVector( MEChannel* ); //JM
  MEVarVector* fillNLSVector( MEChannel*); //JM 



  void rabouteNLS( MEIntervals* intervals, 
		   MEVarVector* varvecin, MEVarVector* varvecout,
		   std::vector<int> ivarin, std::vector<int> ivarout,
		   unsigned int nbuf, unsigned int nave, unsigned int nbad );
  
  void rabouteNLS( METimeInterval* intervals, 
		   MEVarVector* varvecin, MEVarVector* varvecout,
		   std::vector<int> ivarin, std::vector<int> ivarout,
		   unsigned int nbuf, unsigned int nave, unsigned int nbad );


  // Correction variables
  
  int _corVar0;
  int _corVar1;
  int _corZoom0;
  int _corZoom1;
  unsigned _corFitDegree;
  double _corX0;
  double _corY0;
  std::vector<double> _corBeta;
 
  // Create correctors
  
  //virtual void createCorrectors( MEChannel* leaf );
  
  
  // Study correlations

  std::pair<double,double> correlationFwhmMID(MEChannel* leaf_);
  std::vector<double> correlation2Var( MEVarVector* vec0,  MEVarVector* vec1, int var0, int var1, int zoom0, int zoom1, int fitDegree);
  
  
  // Reference key  // TO ADD
  // ==============
  //   METime _refKey[MusEcalHist::iSizeT];


public:

  // declare to ROOT dictionary
  ClassDef(MusEcal,0) // MusEcal -- Monitoring utility for survey of Ecal
};

#endif

