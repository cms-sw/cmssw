#define MELaserPrim_cxx
#include "MELaserPrim.hh"
#include <cassert>
#include <cstdlib>

TString MELaserPrim::apdpn_arrayName[MELaserPrim::iSizeArray_apdpn] = {"APD", "APDoPN", "APDoPNA", "APDoPNB" };
TString MELaserPrim::apdpn_varName[MELaserPrim::iSize_apdpn] = { "Mean", "RMS", "Min", "Max", "Nevt" };
TString MELaserPrim::apdpn_varUnit[MELaserPrim::iSizeArray_apdpn][MELaserPrim::iSize_apdpn] = 
  { { " (ADC Counts)", " (ADC Counts)", " (ADC Counts)", " (ADC Counts)", "" },
    {"", "", "", "", ""},
    {"", "", "", "", ""},
    {"", "", "", "", ""} };
TString MELaserPrim::apdpn_extraVarName[MELaserPrim::iSizeExtra_apdpn] = { "ShapeCor" };
TString MELaserPrim::apdpn_extraVarUnit[MELaserPrim::iSizeExtra_apdpn] = { "" };
TString MELaserPrim::ab_varName[MELaserPrim::iSize_ab] = { "alpha", "beta", "width", "chi2" };
TString MELaserPrim::mtq_varName[MELaserPrim::iSize_mtq] = {"peak", "sigma", "fit", "ampl", "trise", "fwhm", "fw20", "fw80" };
TString MELaserPrim::mtq_varUnit[MELaserPrim::iSize_mtq] = 
  {"(nanoseconds)", "(nanoseconds)", "(nanoseconds)", 
   "(ADC counts)", "(nanoseconds)",
   "(nanoseconds)", "(nanoseconds)", "(nanoseconds)" };
TString MELaserPrim::separator = "__";

MELaserPrim::MELaserPrim(  int dcc, int side, int color, int run, int ts, 
			   const char* inpath, const char* outpath )
  : _dcc(dcc), _side(side), _color(color), _run(run), _ts(ts), _inpath(inpath), _outpath(outpath)
{
  apdpn_file =0;
  ab_file    =0;
  pn_file    =0;
  mtq_file   =0;
  apdpn_tree =0;
  ab_tree    =0;
  pn_tree    =0;
  mtq_tree    =0;
  ixmin      =0;
  ixmax      =0;
  iymin      =0;
  iymax      =0;

  if( _dcc>600 ) _dcc-=600;
  assert( _dcc>0 && _dcc<55 );

  _primStr     = lmfLaserName( iLmfLaserPrim,   _color )+separator;
  _pnPrimStr   = lmfLaserName( iLmfLaserPnPrim, _color )+separator;
  _pulseStr    = lmfLaserName( iLmfLaserPulse,  _color )+separator;

  if( _dcc>=10 && _dcc<=45 )
    {
      _detStr = "EB";
      _sm = _dcc-9;
      if( _sm<19 ) _detStr +=  "+";
      else         _detStr +=  "-";
      
      _sectorStr  = _detStr;
      _sectorStr += "SM";
      if( _sm<10 ) _sectorStr += "0";
      _sectorStr += _sm;
      
    }
  else
    {
      if( _dcc<10 ) 
	{
	  _detStr = "EE-";
	  _sm = _dcc+6;
	}
      else if( _dcc>45 )
	{
	  _detStr = "EE+";
	  _sm = _dcc-45+6;
	}
      if( _sm>9 ) _sm -= 9; 
      
      _sectorStr  = _detStr;
      _sectorStr += "S";
      _sectorStr += _sm;
      
      // special case for sector 5 -> two monitoring regions
      if( _sm!=5 ) _side=0;
    }
  _regionStr  = _sectorStr;
  _regionStr += "-"; _regionStr  += _side;

  //  init();
  //  bookHistograms();
  //fillHistograms();
  //writeHistograms();
}

void 
MELaserPrim::init()
{

  TString cur(_inpath);
  if( !cur.EndsWith("/") ) cur+="/";

  TString _APDPN_fname =cur; _APDPN_fname+="APDPNLaser_Run"; _APDPN_fname+=_run; _APDPN_fname+=".root";
  TString _AB_fname    =cur;    _AB_fname+="AB-PerCrys-Run";    _AB_fname+=_run;    _AB_fname+=".root";
  TString _MTQ_fname   =cur;   _MTQ_fname+="Matacq-Run";       _MTQ_fname+=_run;   _MTQ_fname+=".root";

  apdpn_file = TFile::Open( _APDPN_fname );
  ab_file    = TFile::Open(    _AB_fname );
  pn_file    = apdpn_file;
  mtq_file   = TFile::Open(   _MTQ_fname );

  TString apdpn_tree_name;
  TString    ab_tree_name;
  TString    pn_tree_name;
  TString   mtq_tree_name   = "MatacqShape";

  switch ( _color )
    {
    case iIRed:
      apdpn_tree_name = "Red";
      ab_tree_name    = "Red";
      pn_tree_name    = "RedPN";
      break;
    case iBlue:
      apdpn_tree_name = "Blue";
      ab_tree_name    = "Blue";
      pn_tree_name    = "BluePN";
      break;
    default:
      cout << "MELaserPrim::init() -- Fatal Error -- Wrong Laser Color : " << _color << " ---- Abort " << endl;
      abort();
    }
  
  apdpn_tree = (TTree*) apdpn_file->Get(apdpn_tree_name);
  assert( apdpn_tree!=0 );
  apdpn_tree->SetMakeClass(1);
  apdpn_tree->SetBranchAddress("dccID", &apdpn_dccID, &b_apdpn_dccID);
  apdpn_tree->SetBranchAddress("towerID", &apdpn_towerID, &b_apdpn_towerID);
  apdpn_tree->SetBranchAddress("channelID", &apdpn_channelID, &b_apdpn_channelID);
  apdpn_tree->SetBranchAddress("moduleID", &apdpn_moduleID, &b_apdpn_moduleID);
  apdpn_tree->SetBranchAddress("gainID", &apdpn_gainID, &b_apdpn_gainID);
  apdpn_tree->SetBranchAddress("ieta", &apdpn_ieta, &b_apdpn_ieta);
  apdpn_tree->SetBranchAddress("iphi", &apdpn_iphi, &b_apdpn_iphi);
  apdpn_tree->SetBranchAddress("flag", &apdpn_flag, &b_apdpn_flag);
  apdpn_tree->SetBranchAddress("ShapeCor", &apdpn_ShapeCor, &b_apdpn_ShapeCor);
  for( int jj=0; jj<iSizeArray_apdpn; jj++ )
    {
      // FIXME !!!
      TString name_ = apdpn_arrayName[jj];
      name_.ReplaceAll("PNA","PN0");
      name_.ReplaceAll("PNB","PN1");
      //
      apdpn_tree->SetBranchAddress(name_, apdpn_apdpn[jj], &b_apdpn_apdpn[jj]);
    }

  ab_tree = (TTree*) ab_file->Get(ab_tree_name);
  assert( ab_tree!=0 );
  ab_tree->SetMakeClass(1);
  ab_tree->SetBranchAddress("dccID",     &ab_dccID,     &b_ab_dccID     );
  ab_tree->SetBranchAddress("towerID",   &ab_towerID,   &b_ab_towerID   );
  ab_tree->SetBranchAddress("channelID", &ab_channelID, &b_ab_channelID );
  ab_tree->SetBranchAddress("ieta",      &ab_ieta,      &b_ab_ieta      );
  ab_tree->SetBranchAddress("iphi",      &ab_iphi,      &b_ab_iphi      );
  ab_tree->SetBranchAddress("flag",      &ab_flag,      &b_ab_flag      );
  for( int ii=0; ii<iSize_ab; ii++ )
    {
      ab_tree->SetBranchAddress( ab_varName[ii], &ab_ab[ii], &b_ab_ab[ii] );
    }

  pn_tree = (TTree*) pn_file->Get(pn_tree_name);
  assert( pn_tree!=0 );
  pn_tree->SetMakeClass(1);
  pn_tree->SetBranchAddress( "pnID",     &pn_pnID,     &b_pn_pnID     );
  pn_tree->SetBranchAddress( "moduleID", &pn_moduleID, &b_pn_moduleID );
  pn_tree->SetBranchAddress( "PN",        pn_PN,       &b_pn_PN       );
  pn_tree->SetBranchAddress( "PNoPN",     pn_PNoPN,    &b_pn_PNoPN    );
  pn_tree->SetBranchAddress( "PNoPN0",    pn_PNoPNA,   &b_pn_PNoPNA   );
  pn_tree->SetBranchAddress( "PNoPN1",    pn_PNoPNB,   &b_pn_PNoPNB   );

  mtq_tree = (TTree*) mtq_file->Get(mtq_tree_name);
  assert( mtq_tree!=0 );
  mtq_tree->SetMakeClass(1);
  mtq_tree->SetBranchAddress("event",        &mtq_event,       &b_mtq_event       );
  mtq_tree->SetBranchAddress("laser_color",  &mtq_laser_color, &b_mtq_laser_color );
  mtq_tree->SetBranchAddress("status",       &mtq_status,      &b_mtq_status      );
  for( int ii=0; ii<iSize_mtq; ii++ )
    {
      mtq_tree->SetBranchAddress( mtq_varName[ii],  &mtq_mtq[ii], &b_mtq_mtq[ii] );
    }
}

void
MELaserPrim::bookHistograms()
{
  refresh();

  TString i_name, d_name;
      
  if( _detStr.Contains( "EB" ) )
    {
      ixmin=0;
      ixmax=85;
      nx   =ixmax-ixmin;
      iymin=0;
      iymax=20;
      ny   =iymax-iymin;

      _pn[1] = pair<int,int>(0,5);
      _pn[2] = pair<int,int>(1,6);
      _pn[3] = _pn[2];
      _pn[4] = pair<int,int>(2,7);
      _pn[5] = _pn[4];
      _pn[6] = pair<int,int>(3,8);
      _pn[7] = _pn[6];
      _pn[8] = pair<int,int>(4,9);
      _pn[9] = _pn[8];

    }
  else   // fixme --- to be implemented
    {
      cout << "EE numbering not implemented yet, sorry --- abort" << endl;
      abort();
    }
  
  //
  // Laser ADC Primitives
  //
  bookHistoI( _primStr, "LOGIC_ID" );
  bookHistoI( _primStr, "FLAG" );
  bookHistoF( _primStr, "MEAN" );
  bookHistoF( _primStr, "RMS" );
  bookHistoF( _primStr, "PEAK" );
  bookHistoF( _primStr, "APD_OVER_PNA_MEAN" );
  bookHistoF( _primStr, "APD_OVER_PNA_RMS" );
  bookHistoF( _primStr, "APD_OVER_PNA_PEAK" );
  bookHistoF( _primStr, "APD_OVER_PNB_MEAN" );
  bookHistoF( _primStr, "APD_OVER_PNB_RMS" );
  bookHistoF( _primStr, "APD_OVER_PNB_PEAK" );
  bookHistoF( _primStr, "APD_OVER_PN_MEAN" );
  bookHistoF( _primStr, "APD_OVER_PN_RMS" );
  bookHistoF( _primStr, "APD_OVER_PN_PEAK" );
  bookHistoF( _primStr, "SHAPE_COR" );
  bookHistoF( _primStr, "ALPHA" );
  bookHistoF( _primStr, "BETA" );
  
  TString t_name; 

  //
  // Laser PN Primitives
  //
  t_name = lmfLaserName( iLmfLaserPnPrim, _color );
  addBranchI( t_name, "LOGIC_ID" );
  addBranchI( t_name, "FLAG"     );
  addBranchF( t_name, "MEAN"     );
  addBranchF( t_name, "RMS"      );
  addBranchF( t_name, "PEAK"     );
  addBranchF( t_name, "PNA_OVER_PNB_MEAN"     );
  addBranchF( t_name, "PNA_OVER_PNB_RMS"      );
  addBranchF( t_name, "PNA_OVER_PNB_PEAK"     );

  //
  // Laser Pulse
  //
  t_name = lmfLaserName( iLmfLaserPulse, _color );
  addBranchI( t_name, "LOGIC_ID" );
  addBranchI( t_name, "FIT_METHOD"   );
  addBranchF( t_name, "MTQ_AMPL"     );
  addBranchF( t_name, "MTQ_TIME"     );
  addBranchF( t_name, "MTQ_RISE"     );
  addBranchF( t_name, "MTQ_FWHM"     );
  addBranchF( t_name, "MTQ_FW20"     );
  addBranchF( t_name, "MTQ_FW80"     );
  addBranchF( t_name, "MTQ_SLIDING"  );

  //
  // Laser Config
  //
  t_name = lmfLaserName( iLmfLaserConfig );
  addBranchI( t_name, "LOGIC_ID"       );
  addBranchI( t_name, "WAVELENGTH"     );
  addBranchI( t_name, "VFE_GAIN"       );
  addBranchI( t_name, "PN_GAIN"        );
  addBranchI( t_name, "LSR_POWER"      );
  addBranchI( t_name, "LSR_ATTENUATOR" );
  addBranchI( t_name, "LSR_CURRENT"    );
  addBranchI( t_name, "LSR_DELAY_1"    );
  addBranchI( t_name, "LSR_DELAY_2"    );

  //
  // Laser Run
  //
  t_name = "LMF_RUN_DAT";
  addBranchI( t_name, "LOGIC_ID"       );
  addBranchI( t_name, "NEVENTS"        );
  addBranchI( t_name, "QUALITY_FLAG"   );

  //
  // Laser Run IOV
  //
  t_name = "LMF_RUN_IOV";
  addBranchI( t_name, "TAG_ID"         );
  addBranchI( t_name, "SUB_RUN_NUM"    );
  addBranchI( t_name, "SUB_RUN_START"  );
  addBranchI( t_name, "SUB_RUN_END"    );
  addBranchI( t_name, "DB_TIMESTAMP"   );
  addBranchC( t_name, "SUB_RUN_TYPE"   );

  //
  // Laser LaserRun config dat
  //
  t_name = "RUN_LASERRUN_CONFIG_DAT";
  addBranchI( t_name, "LOGIC_ID"       );
  addBranchC( t_name, "LASER_SEQUENCE_TYPE"    );
  addBranchC( t_name, "LASER_SEQUENCE_COND"  );
}

void
MELaserPrim::fillHistograms()
{
  Long64_t nb       = 0;
  Long64_t nentries = 0;
  Long64_t ientry   = 0;

  nentries = apdpn_tree->GetEntriesFast();
  for( Long64_t jentry=0; jentry<nentries; jentry++ ) 
    {      
      ientry = apdpn_tree->LoadTree( jentry );
      assert( ientry>=0 );
      nb     = apdpn_tree->GetEntry( jentry );

      ientry = ab_tree->LoadTree( jentry );
      assert( ientry>=0 );
      nb     = ab_tree->GetEntry( jentry );

      assert( apdpn_ieta==ab_ieta && apdpn_iphi==ab_iphi );

      int module_ = apdpn_moduleID+1; // fixme  ---  Julie
      int side_ = 1;
      if( module_%2 ) side_=0;

      int ix = apdpn_ieta;
      int iy = apdpn_iphi;

      int id1 = _sm;   // fixme --- this is for barrel
      int id2 = ix*20 + (20 - iy);  // fixme
      int logic_id_ = 1011000000;    // fixme
      logic_id_ += 10000*id1 + id2;

      if( side_!=_side ) 
	{
	  setInt( "LOGIC_ID",           ix, iy,  -logic_id_ );  // fixme --- is this a good idea ?
	  setInt( "FLAG",               ix, iy,  0 );
	  continue;  // fixme --- the other side is pedestal
	}

      int flag = apdpn_flag;

      setInt( "LOGIC_ID",           ix, iy,  logic_id_ );
      setInt( "FLAG",               ix, iy,  flag );
      setVal( "MEAN",               ix, iy,  apdpn_apdpn[iAPD][iMean] );
      setVal( "RMS",                ix, iy,  apdpn_apdpn[iAPD][iRMS] );
      setVal( "PEAK",               ix, iy,  apdpn_apdpn[iAPD][iWbin] );  // fixme --- peak?
      setVal( "APD_OVER_PNA_MEAN",  ix, iy,  apdpn_apdpn[iAPDoPNA][iMean] );
      setVal( "APD_OVER_PNA_RMS",   ix, iy,  apdpn_apdpn[iAPDoPNA][iRMS] );
      setVal( "APD_OVER_PNA_PEAK",  ix, iy,  apdpn_apdpn[iAPDoPNA][iWbin] );  // fixme
      setVal( "APD_OVER_PNB_MEAN",  ix, iy,  apdpn_apdpn[iAPDoPNB][iMean] );
      setVal( "APD_OVER_PNB_RMS",   ix, iy,  apdpn_apdpn[iAPDoPNB][iRMS] );
      setVal( "APD_OVER_PNB_PEAK",  ix, iy,  apdpn_apdpn[iAPDoPNB][iWbin] );  // fixme
      setVal( "APD_OVER_PN_MEAN",   ix, iy,  apdpn_apdpn[iAPDoPN][iMean] );
      setVal( "APD_OVER_PN_RMS",    ix, iy,  apdpn_apdpn[iAPDoPN][iRMS] );
      setVal( "APD_OVER_PN_PEAK",   ix, iy,  apdpn_apdpn[iAPDoPN][iWbin] );  // fixme
      setVal( "SHAPE_COR",          ix, iy,  apdpn_ShapeCor );
      setVal( "ALPHA",              ix, iy,  ab_ab[iAlpha] );
      setVal( "BETA",               ix, iy,  ab_ab[iBeta] );

    }

  TString t_name; 

  //
  // PN primitives
  //
  t_name = lmfLaserName( iLmfLaserPnPrim, _color );

  nentries = pn_tree->GetEntriesFast();
  assert( nentries%2==0 );
  //  cout << "nentries=" << nentries << endl;
  int module_(0), pn_[2];
  int id1_(_sm), id2_(0);
  int logic_id_(0);
  
  Long64_t jentry=0;
  if( _side==1 ) jentry+=2;  // fixme : true also for endcaps?
  while( jentry<nentries ) 
    {      
      for( int jj=0; jj<2; jj++ )
	{
	  // jj=0 --> PNA
	  // jj=1 --> PNB

	  int zentry = jentry+jj;
	  assert( zentry<nentries );
	  
	  ientry = pn_tree->LoadTree( zentry );
	  assert( ientry>=0 );
	  nb     = pn_tree->GetEntry( zentry );

	  if( jj==1 ) assert( pn_moduleID+1==module_ );
	  module_ = pn_moduleID+1;
	  assert( pn_pnID==jj );
	  
	  pn_[jj] = ( jj==0 ) ? _pn[module_].first : _pn[module_].second;
	  id2_ = pn_[jj];
	  logic_id_ = 1131000000 ;
	  //	  logic_id_ = 0;    // fixme
	  logic_id_ += 10000*id1_ + id2_;
	  
 	  i_t[t_name+separator+"LOGIC_ID"] = logic_id_;
 	  f_t[t_name+separator+"MEAN"]  = pn_PN[iMean];
 	  f_t[t_name+separator+"RMS"]   = pn_PN[iRMS];
 	  f_t[t_name+separator+"PEAK"]  = pn_PN[iMean];     // fixme --- peak?
	  f_t[t_name+separator+"PNA_OVER_PNB_MEAN"]  = (jj==0) ? pn_PNoPNB[iMean] : pn_PNoPNA[iMean];
	  f_t[t_name+separator+"PNA_OVER_PNB_RMS" ]  = (jj==0) ? pn_PNoPNB[iRMS]  : pn_PNoPNA[iRMS];
	  f_t[t_name+separator+"PNA_OVER_PNB_PEAK"]  = (jj==0) ? pn_PNoPNB[iMean] : pn_PNoPNA[iMean];   // fixme --- peak?

	  t_t[t_name]->Fill();

	}
      //    cout << "Module=" << module_ << "\tPNA=" << pn_[0] << "\tPNB=" << pn_[1] << endl;

      jentry += 4;
    }

  logic_id_ = 1041000000;
  logic_id_ += 10000*id1_;
  logic_id_ += id1_;
  //
  // MATACQ primitives
  //
  t_name = lmfLaserName( iLmfLaserPulse, _color );
  
  i_t[t_name+separator+"LOGIC_ID"]       = logic_id_;
  i_t[t_name+separator+"FIT_METHOD"]     = _side;   // fixme  --- where is side indicated?
  f_t[t_name+separator+"MTQ_AMPL"]       = 0.;
  f_t[t_name+separator+"MTQ_TIME"]       = 0.;
  f_t[t_name+separator+"MTQ_RISE"]       = 0.;
  f_t[t_name+separator+"MTQ_FWHM"]       = 0.;
  f_t[t_name+separator+"MTQ_FW20"]       = 0.;
  f_t[t_name+separator+"MTQ_FW80"]       = 0.;
  f_t[t_name+separator+"MTQ_SLIDING"]    = 0.;      // fixme  --- sliding: max of average in sliding window
  
  nentries = mtq_tree->GetEntriesFast();
  int nevt   = 0 ;
  int nevent[2] = {0,0} ;
  int side_  = 0  ;
  for( Long64_t jentry=0; jentry<nentries; jentry++ ) 
    {
      ientry = mtq_tree-> LoadTree( jentry );
      assert( ientry>=0 );
      nb = mtq_tree->GetEntry( jentry );

      if( mtq_laser_color!=_color ) continue;
      
      if( mtq_event==1 && nevent[0]>0 )   // fixme --- this is true only now
	{
	  side_   = 1;
	}
      nevent[side_]++;
      if( side_!=_side ) continue;

      bool ok = ( mtq_status==1 ) && ( mtq_mtq[iAmpl]>0 );
      if( !ok ) continue;
      
      nevt++;
      f_t[t_name+separator+"MTQ_AMPL"]       += mtq_mtq[iAmpl];
      f_t[t_name+separator+"MTQ_TIME"]       += mtq_mtq[iPeak];
      f_t[t_name+separator+"MTQ_RISE"]       += mtq_mtq[iTrise];
      f_t[t_name+separator+"MTQ_FWHM"]       += mtq_mtq[iFwhm];
      f_t[t_name+separator+"MTQ_FW20"]       += mtq_mtq[iFw20];
      f_t[t_name+separator+"MTQ_FW80"]       += mtq_mtq[iFw80];
      f_t[t_name+separator+"MTQ_SLIDING"]    += 0.;                 // fix me


   }


  if( nevt!=0 )
    {
      f_t[t_name+separator+"MTQ_AMPL"]       /= nevt;
      f_t[t_name+separator+"MTQ_TIME"]       /= nevt;
      f_t[t_name+separator+"MTQ_RISE"]       /= nevt;
      f_t[t_name+separator+"MTQ_FWHM"]       /= nevt;
      f_t[t_name+separator+"MTQ_FW20"]       /= nevt; 
      f_t[t_name+separator+"MTQ_FW80"]       /= nevt;
      f_t[t_name+separator+"MTQ_SLIDING"]    /= nevt;
    }
  // out << "MTQ -- Laser Color " << _color << " side0= " << nevent[0] << " side1= " << nevent[1] << endl;
  t_t[t_name]->Fill();

  //
  // Laser Run
  //
  t_name = lmfLaserName( iLmfLaserRun );
  //cout << "Fill "<< t_name << endl;
  i_t[t_name+separator+"LOGIC_ID"]       = logic_id_;  // fixme --- is there a channelview for this?
  i_t[t_name+separator+"NEVENTS"]        = nevent[side_];
  i_t[t_name+separator+"QUALITY_FLAG"]   = 1;                // fixme
  t_t[t_name]->Fill();

  //
  // Laser Config
  //
  t_name = lmfLaserName( iLmfLaserConfig );
  //cout << "Fill "<< t_name << endl;
  i_t[t_name+separator+"LOGIC_ID"]        = logic_id_;  // fixme
  i_t[t_name+separator+"WAVELENGTH"]      = _color;
  i_t[t_name+separator+"VFE_GAIN"]        = iVfeGain12; // fixme 
  i_t[t_name+separator+"PN_GAIN"]         = iPnGain16;  // fixme
  i_t[t_name+separator+"LSR_POWER"]       = 0; // will be available from MATACQ data
  i_t[t_name+separator+"LSR_ATTENUATOR"]  = 0; // idem
  i_t[t_name+separator+"LSR_CURRENT"]     = 0; // idem
  i_t[t_name+separator+"LSR_DELAY_1"]     = 0; // idem
  i_t[t_name+separator+"LSR_DELAY_2"]     = 0; // idem
  t_t[t_name]->Fill();

  //
  // Laser Run IOV
  //
  t_name = "LMF_RUN_IOV";
  //cout << "Fill "<< t_name << endl;
  i_t[t_name+separator+"TAG_ID"]        = 0;       // fixme
  i_t[t_name+separator+"SUB_RUN_NUM"]   = _run;      // fixme
  i_t[t_name+separator+"SUB_RUN_START"] = _ts;     // fixme
  i_t[t_name+separator+"SUB_RUN_END"]   = _ts+6; // fixme
  i_t[t_name+separator+"DB_TIMESTAMP"]  = _ts;   //fixme
  c_t[t_name+separator+"SUB_RUN_TYPE"]  = "LASER TEST BEAM H4"; //fixme
  t_t[t_name]->Fill();

//   //
//   // Laser LaserRun config dat
//   //
//   t_name = "RUN_LASERRUN_CONFIG_DAT";
//   //cout << "Fill "<< t_name << endl;
//   i_t[t_name+separator+"LOGIC_ID"]            = logic_id_; //fixme
//   c_t[t_name+separator+"LASER_SEQUENCE_TYPE"] = "LASER TEST BEAM H4"; //fixme
//   c_t[t_name+separator+"LASER_SEQUENCE_COND"] = "SUNNY WEATHER"; //fixme
//   t_t[t_name]->Fill();


}

void
MELaserPrim::writeHistograms()
{
  TString cur(_outpath);
  if( !cur.EndsWith("/") ) cur+="/";

  TString _outfile(cur);
  _outfile += "LMF_";
  _outfile += _regionStr;
  switch( _color )
    {
    case iBlue:   _outfile += "_BlueLaser"; break;
    case iGreen:  _outfile += "_GreenLaser";  break;
    case iRed:    _outfile += "_RedLaser";  break;
    case iIRed:   _outfile += "_IRedLaser";  break;
    default: break;
    }
  _outfile += "_Run";   _outfile += _run;
  _outfile += "_TS";    _outfile += _ts;
  _outfile += ".root";

  out_file = new TFile( _outfile, "RECREATE" );
  //  out_file->cd();


  map< TString, TH2* >::iterator it;

  for( it=i_h.begin(); it!=i_h.end(); it++ )
    {
      it->second->Write();
    }

  for( it=f_h.begin(); it!=f_h.end(); it++ )
    {
      it->second->Write();
    }

  map< TString, TTree* >::iterator it_t;
  for( it_t=t_t.begin(); it_t!=t_t.end(); it_t++ )
    {
      it_t->second->Write();
    }


  out_file->Close();
  delete out_file;
  out_file=0;
}

MELaserPrim::~MELaserPrim()
{
  delete apdpn_tree;
  delete ab_tree;
  delete pn_tree;
  delete mtq_tree;
  if( apdpn_file!=0 )
    {
      apdpn_file->Close();
      delete apdpn_file;
      apdpn_file = 0;
      pn_file=0;
    }
  if( ab_file!=0 )
    {
      ab_file->Close();
      delete ab_file;
      ab_file = 0;
    }
  if( pn_file!=0 )
    {
      pn_file->Close();
      delete pn_file;
      pn_file = 0;
    }
  if( mtq_file!=0 )
    {
      mtq_file->Close();
      delete mtq_file;
      mtq_file = 0;
    }
}

void 
MELaserPrim::print( ostream& o )
{
  o << "DCC/SM/side/color/run/ts " << _dcc << "/" << _sm << "/" << _side << "/" << _color << "/" << _run << "/" << _ts << endl;
  
//   for( int ix=ixmin; ix<ixmax; ix++ )
//     {
//       for( int iy=iymin; iy<iymax; iy++ )
// 	{
// 	  int flag     = getInt( _primStr+"FLAG", ix, iy );
// 	  if( flag==0 ) continue;
// 	  int logic_id = getInt( _primStr+"LOGIC_ID", ix, iy );
// 	  float   apd = getVal( _primStr+"MEAN", ix, iy );
// 	  o << "Crystal ieta=" << ix << "\tiphi=" << iy << "\tlogic_id=" << logic_id << "\tAPD=" << apd <<  endl;
// 	}
//     }
}

TString 
MELaserPrim::lmfLaserName( int table, int color )
{
  TString str("LMF_ERROR");
  if( table<0 || table>=iSizeLmfLaser ) return str;
  if( color<0 || color>=iSizeColor )    return str;

  TString colstr;
  switch( color )
    {
    case iBlue:   colstr = "_BLUE"; break;
    case iGreen:  colstr = "_GREEN"; break;
    case iRed:    colstr =  "_RED";  break;
    case iIRed:   colstr = "_IRED";  break;
    default:  abort();
    }
  str = "LMF_LASER";
  switch( table )
    {
    case iLmfLaserRun:      str  = "LMF_RUN";                 break; 
    case iLmfLaserConfig:   str += "_CONFIG";                 break; 
    case iLmfLaserPulse:    str += colstr; str += "_PULSE";   break; 
    case iLmfLaserPrim:     str += colstr; str += "_PRIM";    break; 
    case iLmfLaserPnPrim:   str += colstr; str += "_PN_PRIM"; break; 
    default:  abort();
    }
  str += "_DAT";
  return str;
}

void
MELaserPrim::addBranchI( const char* t_name_, const char* v_name_ )
{
  TString slashI("/I");
  TString t_name(t_name_);
  TString v_name(v_name_);
  if( t_t.count(t_name)==0 ) t_t[t_name] = new TTree(t_name, t_name);
  t_t[t_name]->Branch(v_name, &i_t[t_name+separator+v_name],v_name+slashI);  
}

void
MELaserPrim::addBranchF( const char* t_name_, const char* v_name_ )
{
  TString slashF("/F");
  TString t_name(t_name_);
  TString v_name(v_name_);
  if( t_t.count(t_name)==0 ) t_t[t_name] = new TTree(t_name, t_name);
  t_t[t_name]->Branch(v_name, &f_t[t_name+separator+v_name],v_name+slashF);
}

void
MELaserPrim::addBranchC( const char* t_name_, const char* v_name_ )
{
  TString slashC("/C");
  TString t_name(t_name_);
  TString v_name(v_name_);
  if( t_t.count(t_name)==0 ) t_t[t_name] = new TTree(t_name, t_name);
  t_t[t_name]->Branch(v_name, &c_t[t_name+separator+v_name],v_name+slashC);
}

void 
MELaserPrim::bookHistoI( const char* h_name_, const char* v_name_ )
{
  TString i_name = TString(h_name_)+TString(v_name_);
  TH2* h_ = new TH2I(i_name,i_name,nx,ixmin,ixmax,ny,iymin,iymax);
  setHistoStyle( h_ );
  i_h[i_name] = h_;
}

void 
MELaserPrim::bookHistoF( const char* h_name_, const char* v_name_ )
{
  TString d_name = TString(h_name_)+TString(v_name_);
  TH2* h_ = new TH2F(d_name,d_name,nx,ixmin,ixmax,ny,iymin,iymax);
  setHistoStyle( h_ );
  f_h[d_name] = h_;
}


bool    
MELaserPrim::setInt( const char* name, int ix, int iy, int ival )
{
  int _ival = getInt( _primStr+name, ix, iy );
  assert( _ival!=-99 );
  if( _ival!=0 ) return false; 

  TH2I* h_ = (TH2I*) i_h[_primStr+name];
  assert( h_!=0 );
  h_->Fill( ix+0.5, iy+0.5, ival );

  return true;
}

bool    
MELaserPrim::setVal( const char* name, int ix, int iy, float val )
{
  float _val = getVal( _primStr+name, ix, iy );
  assert( _val!=-99 );
  if( _val!=0 ) return false; 

  TH2F* h_ = (TH2F*) f_h[_primStr+name];
  assert( h_!=0 );
  h_->Fill( ix+0.5, iy+0.5, val );

  return true;
}

Int_t    
MELaserPrim::getInt( const char* name, int ix, int iy )
{
  Int_t ival=-99;
  if( i_h.count(name)==1 )
    {
      TH2I* h_ = (TH2I*) i_h[name];
      assert( h_!=0 );
      int binx = h_->GetXaxis()->FindBin( ix+0.5 );
      int biny = h_->GetYaxis()->FindBin( iy+0.5 );
      ival =  (Int_t) h_->GetCellContent( binx, biny );
    }
  return ival;
}

Float_t    
MELaserPrim::getVal( const char* name, int ix, int iy )
{
  Float_t val=-99.;
  if( f_h.count(name)==1 )
    {
      TH2F* h_ = (TH2F*) f_h[name];
      assert( h_!=0 );
      int binx = h_->GetXaxis()->FindBin( ix+0.5 );
      int biny = h_->GetYaxis()->FindBin( iy+0.5 );
      val =  h_->GetCellContent( binx, biny );
    }
  return val;
}

bool
MELaserPrim::setInt( const char* tname, const char* vname, int ival )
{
  TString key_(tname); key_ += separator; key_ += vname;
  assert( i_t.count(key_)==1 );
  i_t[key_] = ival;
  return true;
}

bool
MELaserPrim::setVal( const char* tname, const char* vname, float val )
{
  TString key_(tname); key_ += separator; key_ += vname;
  assert( f_t.count(key_)==1 );
  f_t[key_]  = val;
  return true;
}

bool
MELaserPrim::fill( const char* tname )
{
  TString key_( tname );
  assert( t_t.count(key_)==1 );
  t_t[key_] -> Fill();
  return true;
}

void 
MELaserPrim::setHistoStyle( TH1* h )
{
  if( h==0 ) return;
  
  float _scale = 1;

  h->SetLineColor(4);
  h->SetLineWidth(1);
  h->SetFillColor(38);
  TAxis* axis[3];
  axis[0] = h->GetXaxis();
  axis[1] = h->GetYaxis();
  axis[2] = h->GetZaxis();
  for( int ii=0; ii<3; ii++ )
    {
      TAxis* a = axis[ii];
      if( !a ) continue;
      a->SetLabelFont(132);
      a->SetLabelOffset(_scale*0.005);
      a->SetLabelSize(_scale*0.04);
      a->SetTitleFont(132);
      a->SetTitleOffset(_scale*1);
      a->SetTitleSize(_scale*0.04);
    } 
  h->SetStats( kTRUE );
}

void
MELaserPrim::refresh()
{
  map< TString, TH2* >::iterator it;

  for( it=i_h.begin(); it!=i_h.end(); it++ )
    {
      delete it->second;
      it->second = 0;
    }
  i_h.clear();

  for( it=f_h.begin(); it!=f_h.end(); it++ )
    {
      delete it->second;
      it->second = 0;
    }
  f_h.clear();

  map< TString, TTree* >::iterator it_t;
  for( it_t=t_t.begin(); it_t!=t_t.end(); it_t++ )
    {
      delete it_t->second;
      it->second = 0;
    }
  t_t.clear();
}
