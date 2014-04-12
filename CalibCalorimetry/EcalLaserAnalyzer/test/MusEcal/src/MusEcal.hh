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

// #include "MEIntervals.hh"
// #include "MECorrector2Var.hh"

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
  enum { iNLS, iCorNLS, iAPDoPNA, iAPDoPNB, iAPDoPN, iAPD, iAPDTime, 
	 iPNA, iPNB, iPNBoPNA, iAlphaBeta, iAlphaBeta_used, iShapeCor, 
	 iMTQTrise, iMTQAmpl, iMTQFwhm, iMTQFw20, iMTQFw80, iMTQTime,
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
  static int firstRun;
  static int lastRun;

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
  void fillHistograms();
  void fillEBGlobalHistograms();
  void fillEBLocalHistograms();
  void fillEEGlobalHistograms();
  void fillEELocalHistograms();
  void writeGlobalHistograms();

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

//   // validity intervals
//   void setMtqVar( int mtqVar0, int mtqVar1, int mtqLevel );
//   void setPnVar(  int  pnVar0, int  pnVar1, int  pnLevel );

//   MEIntervals* mtqIntervals( int type );
//   MEKeyInterval* topMtqInterval( int type ) { return mtqIntervals(type)->topInterval(); }

//   MEIntervals* pnIntervals( int type );
//   MEKeyInterval* topPnInterval( int type ) { return pnIntervals(type)->topInterval(); }
//   std::map< MEKeyInterval*, int >& choices( int type ) { return _choices[type]; }
 
//   // create corrections
//   virtual void createCorrectors();
  
//   // apply corrections
//   void applyCorrections( MELeaf* leaf, int type, int var0, int var1, 
// 			 std::vector< MERunKey>& keys, std::vector< MERunKey>& altkeys,
// 			 std::vector< double >& val, std::vector< double >& cor_val, std::vector< double >& glued_val, std::vector< double >& cor_and_glued_val,
// 			 std::vector< bool   >&  ok, std::vector< bool   >& cor_ok , std::vector< bool   >& glued_ok,  std::vector< bool   >& cor_and_glued_ok  );

//   void rabouteVector( MEKeyInterval*, int level, const std::vector< MERunKey >& keys, std::vector< double >& val, std::vector< bool >& ok, 
// 		      MERunKey curKey=0, unsigned nbuf=2, unsigned nave=10 );
//   void rabouteVector( MEKeyInterval*, int level, MEVarMap& val, MEBoolMap& ok, 
// 		      MERunKey curKey=0, unsigned nbuf=2, unsigned nave=10 );
  
//   void dumpHistory();

//   // current variable
//   int _var;
  
//   // current zoom
//   int _zoom;

//   // current channel of group of channels
//   bool _newVar;
//   bool _newSelection;
//   bool _newRange;
//   bool _newReference;
//   bool _iSelect;

//   // validity intervals the Side level, based on MATACQ variables
//   bool _mtqTwoVar;
//   int  _mtqLevel;
//   int  _mtqVar0;
//   int  _mtqVar1;

//   // validity intervals at the Module level, based on the PNA and PNB signal variations
//   bool _pnTwoVar;
//   int  _pnLevel;
//   int  _pnVar0;
//   int  _pnVar1;

//   // variable to correct against
//   int _corVar0;
//   int _corVar1;
//   int _corZoom0;
//   int _corZoom1;
//   unsigned _corFitDegree;
//   double _corX0;
//   double _corY0;
//   std::vector< double > _corBeta;
 
//   // intervals, as a function of the Leaf (at the Side or Module level)
//   std::map< MELeaf*, MEIntervals* > _intervals[MusEcalHist::iSizeT];

//   // choice of variables in PN intervals
//   std::map< MEKeyInterval*, int > _choices[MusEcalHist::iSizeT];

  
//   // range range and normalisation
//   bool _normalize;

//   // reference key
//   MERunKey _refKey[MusEcalHist::iSizeT];


  
//   virtual void buildMtqIntervals( int type );
//   virtual void buildPnIntervals( int type );

public:

  // declare to ROOT dictionary
  ClassDef(MusEcal,0) // MusEcal -- Monitoring utility for survey of Ecal
};

#endif

