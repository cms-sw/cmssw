#ifndef MELaserPrim_hh
#define MELaserPrim_hh

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

#include <TROOT.h>
#include <TChain.h>
#include <TH2I.h>
#include <TH2F.h>
#include <TFile.h>
#include <TString.h>

#include "ME.h"

class MELaserPrim 
{

public:

  // Julie's ntuple variables
  enum { iAPD, iAPDoPN, iAPDoPNA, iAPDoPNB, iAPDoAPD, iAPDoAPDA, iAPDoAPDB, iTime, iSizeArray_apdpn }; 
  enum { iMean, iRMS, iM3, iNevt, iMin, iMax, iSize_apdpn }; 
  enum { iShapeCor, iSizeExtra_apdpn };
  enum { iAlpha, iBeta, iWidth, iChi2, iSize_ab };
  enum { iPeak, iSigma, iFit, iAmpl, iTrise, iFwhm, iFw20, iFw80, iSlide, iSize_mtq };

  enum { iGain0, iGain1, iGain2, iGain3, iSize_gain }; 

  // channel views & logic ids
  enum { iECAL, iECAL_LMR, 
	 iEB_crystal_number, iEB_LM_LMM, iEB_LM_PN,
	 iEE_crystal_number, iEE_LM_LMM, iEE_LM_PN, iSize_cv };
  static TString channelViewName( int );
  static int logicId( int channelView, int id1, int id2=0 );
  static bool getViewIds( int logicId, int& channelView, int& id1, int& id2 );
 
  // constructor
  MELaserPrim( ME::Header header, ME::Settings settings, 
	       const char* inpath, const char* outfile ); 

  // destructor
  virtual ~MELaserPrim();

  // functions
  bool init_ok;
  void init();
  void bookHistograms();
  void fillHistograms();
  void writeHistograms();
  void print(std::ostream& o );

  // name of tables
  static TString lmfLaserName( int table, int type,  
			       int color=ME::iBlue );

  // fill histograms from a user application (e.g. DB )
  bool setInt( const char*, int ix, int iy,   int ival );
  bool setVal( const char*, int ix, int iy, float  val );

  // fill tree variables from a user application (e.g. DB )
  bool setInt( const char* tname, const char* vname, int ival );
  bool setVal( const char* tname, const char* vname, float  val );

  bool fill( const char* tname );

  // access
  Int_t    getInt( const char*, int ix, int iy );
  Float_t  getVal( const char*, int ix, int iy );

  static TString separator;

private:

  // monitoring region (dcc+side), wavelength, run number and timestamp
  int   _reg;
  bool  _isBarrel;
  int   _lmr;
  int   _dcc;
  int   _side;
  int   _run;
  int   _lb;
  int   _events;

  int _type;
  int _color;
  int _power;
  int _filter;
  int _delay;

  // GHM
  ME::TimeStamp    _ts;
  ME::TimeStamp    _ts_beg;
  ME::TimeStamp    _ts_end;

  int    _mgpagain; 
  int    _memgain; 

  // useful 
  int  _ecal_region;
  int  _sm;
  //  std::map< int, std::pair<int, int> > _pn;  // association module -> pair of PN diodes

  TString _sectorStr;
  TString _regionStr;
  TString _primStr;
  TString _pnPrimStr;
  TString _pulseStr;
  TString _tpPrimStr;
  TString _tpPnPrimStr;

  // root files
  TFile* apdpn_file;
  TFile*    ab_file;
  TFile*    pn_file;
  TFile*   mtq_file;
  TFile* tpapd_file;
  TFile*   out_file;

  // root trees
  TTree* apdpn_tree;
  TTree*    ab_tree;
  TTree*    pn_tree;
  TTree*   mtq_tree;  
  TTree* tpapd_tree;
  TTree*  tppn_tree;
  
  // paths to laser monitoring trees
  TString  _inpath;

  // root file in output
  TString _outfile;

  // index limits (depends on dcc)
  int nx;
  int ixmin;
  int ixmax;
  int ny;
  int iymin;
  int iymax;

  // 2D-histograms
  std::map< TString, TH2* > i_h;    // integer quantities
  std::map< TString, TH2* > f_h;    // floating point quantities

  // trees
  std::map< TString, TTree* >  t_t;  // map of trees
  std::map< TString, int >     i_t;  // integer values
  std::map< TString, float >   f_t;  // float values
  std::map< TString, const char* > c_t;  // string values
  
  // leaves for the APDPN ntuple
  Int_t           apdpn_dccID;
  Int_t           apdpn_towerID;
  Int_t           apdpn_channelID;
  Int_t           apdpn_moduleID;
  //  Double_t        apdpn_gainID;
  Int_t           apdpn_side;
  Int_t           apdpn_ieta;
  Int_t           apdpn_iphi;
  Int_t           apdpn_flag;
  Double_t        apdpn_ShapeCor;
  Double_t        apdpn_apdpn[iSizeArray_apdpn][iSize_apdpn];

  // leaves for the AB ntuple
  Int_t           ab_dccID;
  Int_t           ab_towerID;
  Int_t           ab_channelID;
  Int_t           ab_ieta;
  Int_t           ab_iphi;
  Int_t           ab_flag;
  Double_t        ab_ab[iSize_ab];

  // leaves for the PN ntuple
  Int_t           pn_side;
  Int_t           pn_pnID;
  Int_t           pn_moduleID;
  Double_t        pn_PN[iSize_apdpn];
  Double_t        pn_PNoPN[iSize_apdpn];
  Double_t        pn_PNoPNA[iSize_apdpn];
  Double_t        pn_PNoPNB[iSize_apdpn];

  // leaves for the MTQ ntuple
  Int_t           mtq_side;
  Int_t           mtq_color;
  Double_t        mtq_mtq[iSize_mtq];

  // leaves for the TPAPD ntuple
  Int_t           tpapd_iphi;
  Int_t           tpapd_ieta;
  Int_t           tpapd_dccID;
  Int_t           tpapd_side;
  Int_t           tpapd_towerID;
  Int_t           tpapd_channelID;
  Int_t           tpapd_moduleID;
  Int_t           tpapd_flag;
  Int_t           tpapd_gain;
  Double_t        tpapd_APD[iSize_apdpn];  

  // leaves for the TPPN ntuple
  Int_t           tppn_side;
  Int_t           tppn_pnID;
  Int_t           tppn_moduleID;
  Int_t           tppn_gain;
  Double_t        tppn_PN[iSize_apdpn];

  // List of branches for APDPN
  TBranch        *b_apdpn_dccID;   //!
  TBranch        *b_apdpn_towerID;   //!
  TBranch        *b_apdpn_channelID;   //!
  TBranch        *b_apdpn_moduleID;   //!
  //  TBranch        *b_apdpn_gainID;   //!
  TBranch        *b_apdpn_side;   //!
  TBranch        *b_apdpn_ieta;   //!
  TBranch        *b_apdpn_iphi;   //!
  TBranch        *b_apdpn_flag;   //!
  TBranch        *b_apdpn_ShapeCor;   //!
  TBranch        *b_apdpn_apdpn[iSizeArray_apdpn];   //!

  // List of branches for AB
  TBranch        *b_ab_dccID;   //!
  TBranch        *b_ab_towerID;   //!
  TBranch        *b_ab_channelID;   //!
  TBranch        *b_ab_ieta;   //!
  TBranch        *b_ab_iphi;   //!
  TBranch        *b_ab_flag;   //!
  TBranch        *b_ab_ab[iSize_ab];   //!

  // List of branches for PN
  TBranch        *b_pn_side;   //!
  TBranch        *b_pn_pnID;   //!
  TBranch        *b_pn_moduleID;   //!
  TBranch        *b_pn_PN;   //!
  TBranch        *b_pn_PNoPN;   //!
  TBranch        *b_pn_PNoPNA;   //!
  TBranch        *b_pn_PNoPNB;   //!

  // List of branches for MTQ
  TBranch        *b_mtq_side;   //!
  TBranch        *b_mtq_color;   //!
  TBranch        *b_mtq_mtq[iSize_mtq]; //!

  // List of branches for TPAPD
  TBranch        *b_tpapd_iphi;   //!
  TBranch        *b_tpapd_ieta;   //!
  TBranch        *b_tpapd_dccID;   //!
  TBranch        *b_tpapd_side;   //!
  TBranch        *b_tpapd_towerID;   //!
  TBranch        *b_tpapd_channelID;   //!
  TBranch        *b_tpapd_moduleID;   //!
  TBranch        *b_tpapd_flag;   //!
  TBranch        *b_tpapd_gain;   //!
  TBranch        *b_tpapd_APD;   //!

  // List of branches for TPPN
  TBranch        *b_tppn_side;   //!
  TBranch        *b_tppn_pnID;   //!
  TBranch        *b_tppn_moduleID;   //!
  TBranch        *b_tppn_gain;   //!
  TBranch        *b_tppn_PN;   //!
  
  static TString apdpn_arrayName[iSizeArray_apdpn];
  static TString apdpn_varName[iSize_apdpn];
  static TString apdpn_varUnit[iSizeArray_apdpn][iSize_apdpn];
  static TString apdpn_extraVarName[iSizeExtra_apdpn];
  static TString apdpn_extraVarUnit[iSizeExtra_apdpn];
  static TString ab_varName[iSize_ab];
  static TString mtq_varName[iSize_mtq];
  static TString mtq_varUnit[iSize_mtq];

  void setHistoStyle( TH1* );
  void refresh();
  void addBranchI( const char* t_name, const char* v_name );
  void addBranchF( const char* t_name, const char* v_name );
  void addBranchC( const char* t_name, const char* v_name );
  void bookHistoI( const char* t_name, const char* v_name );
  void bookHistoF( const char* t_name, const char* v_name );

  //  ClassDef( MELaserPrim, 0 ) // reads/writes Laser primitives

};

#endif
