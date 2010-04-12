#ifndef MENLS_hh
#define MENLS_hh

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

using namespace std;

#include <TROOT.h>
#include <TChain.h>
#include <TH2I.h>
#include <TH2F.h>
#include <TFile.h>
#include <TString.h>

#include "ME.h"

class MENLS 
{

public:

  // Julie's ntuple variables 
  enum { iMean, iRMS, iNevt, iFlag, iNorm, iENorm, iSize_nls }; 
  enum { iRunRef, iLBRef, iTimeLowRef, iTimeHighRef, iSizeRef_nls }; 
 
  // channel views & logic ids
  enum { iECAL, iECAL_LMR, 
	 iEB_crystal_number, iEB_LM_LMM, iEB_LM_PN,
	 iEE_crystal_number, iEE_LM_LMM, iEE_LM_PN, iSize_cv };

  static TString channelViewName( int );
  static int logicId( int channelView, int id1, int id2=0 );
  static bool getViewIds( int logicId, int& channelView, int& id1, int& id2 );
 
  // constructor
  MENLS( ME::Header header, ME::Settings settings, 
	       const char* outfile ); 

  // destructor
  virtual ~MENLS();

  // functions
  bool init_ok;
  void init();
  void bookHistograms();
  void fillRefChannel(int ieta, int iphi, int runnum, int lbnum, int startlow, int starthigh );
  void fillChannel(int ieta, int iphi, int ivar, float val );
  void fillRun();
  void writeHistograms();
  void print( ostream& o );

  // name of tables
  static TString lmfNLSName( int table, int color=ME::iBlue );
  static TString nlsVarName(int var);
  static TString nlsRefVarName(int var);

  // fill histograms from a user application (e.g. DB )
  bool setInt( const char*, int ix, int iy,   int ival );
  bool setVal( const char*, int ix, int iy, float  val );
  // fill histograms from a user application (e.g. DB )
  bool setIntRef( const char*, int ix, int iy,   int ival );
  bool setValRef( const char*, int ix, int iy, float  val );

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

  bool _debug;

  // GHM
  ME::TimeStamp    _ts;
  ME::TimeStamp    _ts_beg;
  ME::TimeStamp    _ts_end;

  int    _mgpagain; 
  int    _memgain; 

  // useful 
  int  _ecal_region;
  int  _sm;
  //  map< int, pair<int, int> > _pn;  // association module -> pair of PN diodes

  TString _sectorStr;
  TString _regionStr;
  TString _nlsStr;
  TString _refStr;

  TFile*   out_file;

  
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
  map< TString, TH2* > i_h;    // integer quantities
  map< TString, TH2* > f_h;    // floating point quantities

  // trees
  map< TString, TTree* >  t_t;  // map of trees
  map< TString, int >     i_t;  // integer values
  map< TString, float >   f_t;  // float values
  map< TString, const char* > c_t;  // string values

  void setHistoStyle( TH1* );
  void refresh();
  void addBranchI( const char* t_name, const char* v_name );
  void addBranchF( const char* t_name, const char* v_name );
  void addBranchC( const char* t_name, const char* v_name );
  void bookHistoI( const char* t_name, const char* v_name );
  void bookHistoF( const char* t_name, const char* v_name );

  static TString nls_histName[iSize_nls];
  static TString nlsref_histName[iSizeRef_nls];
  //  ClassDef( MENLS, 0 ) // reads/writes Laser primitives

};

#endif
