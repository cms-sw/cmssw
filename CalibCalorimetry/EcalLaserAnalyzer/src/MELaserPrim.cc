#define MELaserPrim_cxx
#include <assert.h>
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MELaserPrim.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEGeom.h"
#include <cassert>
#include <cstdlib>

TString MELaserPrim::apdpn_arrayName[MELaserPrim::iSizeArray_apdpn] = {"APD", "APDoPN", "APDoPNA", "APDoPNB","APDoAPD","APDoAPDA", "APDoAPDB", "Time"};
TString MELaserPrim::apdpn_varName[MELaserPrim::iSize_apdpn] = { "Mean", "RMS", "M3", "Nevt", "Min", "Max"};
TString MELaserPrim::apdpn_varUnit[MELaserPrim::iSizeArray_apdpn][MELaserPrim::iSize_apdpn] = 
 
  { { " (ADC Counts)", " (ADC Counts)", " (ADC Counts)" ,"", " (ADC Counts)", " (ADC Counts)"},
    {"", "", "", "", "", ""},
    {"", "", "", "", "", ""},
    {"", "", "", "", "", ""},
    {"", "", "", "", "", ""},
    {"", "", "", "", "", ""},
    {" (25 ns)", " (25 ns)", " (25 ns)", "", " (25 ns)", " (25 ns)"} };
TString MELaserPrim::apdpn_extraVarName[MELaserPrim::iSizeExtra_apdpn] = { "ShapeCor" };
TString MELaserPrim::apdpn_extraVarUnit[MELaserPrim::iSizeExtra_apdpn] = { "" };
TString MELaserPrim::ab_varName[MELaserPrim::iSize_ab] = { "alpha", "beta", "width", "chi2" };
TString MELaserPrim::mtq_varName[MELaserPrim::iSize_mtq] = {"peak", "sigma", "fit", "ampl", "trise", "fwhm", "fw20", "fw80", "sliding" };
TString MELaserPrim::mtq_varUnit[MELaserPrim::iSize_mtq] = 
  {"(nanoseconds)", "(nanoseconds)", "(nanoseconds)", 
   "(ADC counts)", "(nanoseconds)",
   "(nanoseconds)", "(nanoseconds)", "(nanoseconds)", "(ADC counts)"};

TString MELaserPrim::separator = "__";

//ClassImp( MELaserPrim )

MELaserPrim::MELaserPrim(  ME::Header header, ME::Settings settings, 
			   const char* inpath, const char* outfile )
: init_ok(false), _isBarrel(true), _inpath(inpath), _outfile(outfile)
{
  apdpn_file =0;
  ab_file    =0;
  mtq_file   =0;
  tpapd_file =0;
  apdpn_tree =0;
  ab_tree    =0;
  pn_tree    =0;
  mtq_tree   =0;
  tpapd_tree =0;
  tppn_tree  =0;
  ixmin      =0;
  ixmax      =0;
  iymin      =0;
  iymax      =0;

  _dcc    = header.dcc;
  _side   = header.side;
  _run    = header.run;
  _lb     = header.lb;
  _events = header.events;
  _ts     = header.ts_beg;
  _ts_beg = header.ts_beg;
  _ts_end = header.ts_end;

  _type     = settings.type;
  _color    = settings.wavelength;
  _power    = settings.power;
  _filter   = settings.filter;
  _delay    = settings.delay;
  _mgpagain = settings.mgpagain; 
  _memgain  = settings.memgain; 

  if( _type==ME::iLaser )
    {
      _primStr     = lmfLaserName( ME::iLmfLaserPrim,   _type, _color )+separator;
      _pnPrimStr   = lmfLaserName( ME::iLmfLaserPnPrim, _type, _color )+separator;
      _pulseStr    = lmfLaserName( ME::iLmfLaserPulse,  _type, _color )+separator;
    }
  else if( _type==ME::iTestPulse )
    {
      _tpPrimStr    = lmfLaserName( ME::iLmfTestPulsePrim, _type   )+separator;
      _tpPnPrimStr  = lmfLaserName( ME::iLmfTestPulsePnPrim, _type )+separator;
    }

  _lmr = ME::lmr( _dcc, _side );
  ME::regionAndSector( _lmr, _reg, _sm, _dcc, _side );    
  _isBarrel = (_reg==ME::iEBM || _reg==ME::iEBP);
  _sectorStr  = ME::smName( _lmr );
  _regionStr  = _sectorStr;
  _regionStr += "_"; _regionStr  += _side;

  init();
  bookHistograms();
  //fillHistograms();
  //writeHistograms();
}

TString
MELaserPrim::channelViewName( int iname )
{
  switch( iname )
    {
    case iECAL:                 return "ECAL";
    case iECAL_LMR:             return "ECAL_LMR";
    case iEB_crystal_number:    return "EB_crystal_number";
    case iEB_LM_LMM:            return "EB_LM_LMM";
    case iEB_LM_PN:             return "EB_LM_PN";
    case iEE_crystal_number:    return "EE_crystal_number";
    case iEE_LM_LMM:            return "EE_LM_LMM";
    case iEE_LM_PN:             return "EE_LM_PN";
    default:
      abort();
    }
  return "";
}

int
MELaserPrim::logicId( int channelView, int id1, int id2 )
{
  assert( channelView>=iECAL && channelView<iSize_cv );
  return 1000000*channelView + 10000*id1 + id2;
}

bool 
MELaserPrim::getViewIds( int logic_id, int& channelView, int& id1, int& id2 )
{
  bool out = true;
  int channelView_ = logic_id/1000000;
  if( channelView!=0 && channelView_!=channelView ) out=false;
  channelView = channelView_;
  id1 = (logic_id%1000000)/10000;
  id2 = logic_id%10000;
  return out;
}

void 
MELaserPrim::init()
{
  bool verbose_ = false;

  if( _inpath=="0" )
    {
      if( verbose_ ) std::cout << "no input file" << std::endl;
      init_ok = true;
      return; // GHM
    }

  TString cur(_inpath);
  if( !cur.EndsWith("/") ) cur+="/";

  if( _type==ME::iLaser )
    {
      TString _APDPN_fname =cur; _APDPN_fname += "APDPN_LASER.root";
      TString _AB_fname    =cur; _AB_fname    += "AB.root"; 
      TString _MTQ_fname   =cur; _MTQ_fname   += "MATACQ.root";

      bool apdpn_ok, ab_ok, pn_ok, mtq_ok;
      apdpn_ok=false; ab_ok=false; pn_ok=false; mtq_ok=false;

      FILE *test; 
      test = fopen( _APDPN_fname,"r");  
      if (test) {
	apdpn_ok = true;
	pn_ok = true;
	fclose( test );
      }
      test = fopen( _AB_fname,"r");  
      if (test) 
	{
	  ab_ok = true;
	  fclose( test );
	}
      test = fopen( _MTQ_fname,"r");  
      if (test) 
	{
	  mtq_ok = true;
	  fclose( test );
	}

      if(apdpn_ok) apdpn_file = TFile::Open( _APDPN_fname );
      if(ab_ok)    ab_file    = TFile::Open(    _AB_fname );
      if(mtq_ok)  mtq_file   = TFile::Open(   _MTQ_fname );

      if( verbose_ )
	{
	  std::cout << _APDPN_fname << " ok=" << apdpn_ok << std::endl;
	  std::cout << _AB_fname    << " ok=" << ab_ok    << std::endl;
	  std::cout << _MTQ_fname   << " ok=" << mtq_ok    << std::endl;
	}
      if (!apdpn_ok || !pn_ok ) return; // FIXME !
  
      TString apdpn_tree_name;
      TString    ab_tree_name;
      TString    pn_tree_name;
      TString   mtq_tree_name; 

      apdpn_tree_name = "APDCol";
      ab_tree_name    = "ABCol";
      pn_tree_name    = "PNCol";   
      mtq_tree_name   = "MatacqCol";
  
      apdpn_tree_name += _color;
      ab_tree_name    += _color;
      pn_tree_name    += _color;
      mtq_tree_name   += _color;

      if(mtq_ok)  {
	TTree *ckeckMtq = (TTree*) mtq_file->Get(mtq_tree_name);
	if( ckeckMtq ==0 ) mtq_ok = false;
      }

      if( _color != ME::iIRed && _color != ME::iBlue ){ 
	std::cout << "MELaserPrim::init() -- Fatal Error -- Wrong Laser Color : " << _color << " ---- Abort " << std::endl;
	return;
      }
      
      apdpn_tree = (TTree*) apdpn_file->Get(apdpn_tree_name);
      assert( apdpn_tree!=0 );
      apdpn_tree->SetMakeClass(1);
      apdpn_tree->SetBranchAddress("dccID", &apdpn_dccID, &b_apdpn_dccID);
      apdpn_tree->SetBranchAddress("towerID", &apdpn_towerID, &b_apdpn_towerID);
      apdpn_tree->SetBranchAddress("channelID", &apdpn_channelID, &b_apdpn_channelID);
      apdpn_tree->SetBranchAddress("moduleID", &apdpn_moduleID, &b_apdpn_moduleID);
      apdpn_tree->SetBranchAddress("side", &apdpn_side, &b_apdpn_side);
      apdpn_tree->SetBranchAddress("ieta", &apdpn_ieta, &b_apdpn_ieta);
      apdpn_tree->SetBranchAddress("iphi", &apdpn_iphi, &b_apdpn_iphi);
      apdpn_tree->SetBranchAddress("flag", &apdpn_flag, &b_apdpn_flag);
      if( apdpn_tree->GetBranchStatus("ShapeCor")) apdpn_tree->SetBranchAddress("ShapeCor", &apdpn_ShapeCor, &b_apdpn_ShapeCor);
      else apdpn_ShapeCor = 0.0;
      for( int jj=0; jj<iSizeArray_apdpn; jj++ )
	{
	  TString name_ = apdpn_arrayName[jj];
	  apdpn_tree->SetBranchAddress(name_, apdpn_apdpn[jj], &b_apdpn_apdpn[jj]);
	}
  
      if(ab_ok)  {
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
      }

      pn_tree = (TTree*) apdpn_file->Get(pn_tree_name);
      assert( pn_tree!=0 );
      pn_tree->SetMakeClass(1);
      pn_tree->SetBranchAddress( "side",     &pn_side,     &b_pn_side     );
      pn_tree->SetBranchAddress( "pnID",     &pn_pnID,     &b_pn_pnID     );
      pn_tree->SetBranchAddress( "moduleID", &pn_moduleID, &b_pn_moduleID );
      pn_tree->SetBranchAddress( "PN",        pn_PN,       &b_pn_PN       );
      pn_tree->SetBranchAddress( "PNoPN",     pn_PNoPN,    &b_pn_PNoPN    );
      pn_tree->SetBranchAddress( "PNoPNA",    pn_PNoPNA,   &b_pn_PNoPNA   );
      pn_tree->SetBranchAddress( "PNoPNB",    pn_PNoPNB,   &b_pn_PNoPNB   );


      if( mtq_ok ) {
	mtq_tree = (TTree*) mtq_file->Get(mtq_tree_name);
	assert( mtq_tree!=0 );
	mtq_tree->SetMakeClass(1);
	mtq_tree->SetBranchAddress("side",        &mtq_side,       &b_mtq_side  );
    
	for( int ii=0; ii<iSize_mtq; ii++ )
	  {
	    mtq_tree->SetBranchAddress( mtq_varName[ii],  &mtq_mtq[ii], &b_mtq_mtq[ii] );
	  }
      }
    }
  else if( _type==ME::iTestPulse )
    {
      TString _TPAPD_fname =cur; _TPAPD_fname += "APDPN_TESTPULSE.root";

      bool tpapd_ok;
      tpapd_ok=false; 

      FILE *test; 
      test = fopen( _TPAPD_fname,"r");  
      if (test) {
	tpapd_ok = true;
	fclose( test );
      }

      if(tpapd_ok) tpapd_file = TFile::Open( _TPAPD_fname );

      if( verbose_ )
	{
	  std::cout << _TPAPD_fname << " ok=" << tpapd_ok << std::endl;
	}
      if (!tpapd_ok ) return;
  
      TString tpapd_tree_name;
      TString tppn_tree_name;

      tpapd_tree_name = "TPAPD";
      tppn_tree_name  = "TPPN";   
    
      tpapd_tree = (TTree*) tpapd_file->Get(tpapd_tree_name);
      assert( tpapd_tree!=0 );
      tpapd_tree->SetMakeClass(1);
      tpapd_tree->SetBranchAddress("ieta", &tpapd_ieta, &b_tpapd_ieta);
      tpapd_tree->SetBranchAddress("iphi", &tpapd_iphi, &b_tpapd_iphi);
      tpapd_tree->SetBranchAddress("dccID", &tpapd_dccID, &b_tpapd_dccID);
      tpapd_tree->SetBranchAddress("side", &tpapd_side, &b_tpapd_side);
      tpapd_tree->SetBranchAddress("towerID", &tpapd_towerID, &b_tpapd_towerID);
      tpapd_tree->SetBranchAddress("channelID", &tpapd_channelID, &b_tpapd_channelID);
      tpapd_tree->SetBranchAddress("moduleID", &tpapd_moduleID, &b_tpapd_moduleID);
      tpapd_tree->SetBranchAddress("flag", &tpapd_flag, &b_tpapd_flag);
      tpapd_tree->SetBranchAddress("gain", &tpapd_gain, &b_tpapd_gain);
      tpapd_tree->SetBranchAddress("APD",   tpapd_APD,  &b_tpapd_APD );

      tppn_tree = (TTree*) tpapd_file->Get(tppn_tree_name);
      assert( tppn_tree!=0 );
      tppn_tree->SetMakeClass(1);
      tppn_tree->SetBranchAddress( "side",     &tppn_side,     &b_tppn_side     );
      tppn_tree->SetBranchAddress( "pnID",     &tppn_pnID,     &b_tppn_pnID     );
      tppn_tree->SetBranchAddress( "moduleID", &tppn_moduleID, &b_tppn_moduleID );
      tppn_tree->SetBranchAddress( "gain",     &tppn_gain,     &b_tppn_gain     );
      tppn_tree->SetBranchAddress( "PN",        tppn_PN,       &b_tppn_PN       );
    }
  init_ok = true;
}

void
MELaserPrim::bookHistograms()
{
  if( !init_ok ) return;
  refresh();

  TString i_name, d_name;
      
  if( _isBarrel )
    {
      ixmin=0;
      ixmax=85;
      nx   =ixmax-ixmin;
      iymin=0;
      iymax=20;
      ny   =iymax-iymin;

//       for( int ilmod=1; ilmod<=9; ilmod++ )
// 	{
// 	  _pn[ilmod] = MEEBGeom::pn( ilmod );
// 	}
    }
  else   // fixme --- to be implemented
    {
      ixmin=1;
      ixmax=101;
      nx   =ixmax-ixmin;
      iymin=1;
      iymax=101;
      ny   =iymax-iymin;
//       for( int ilmod=1; ilmod<=21; ilmod++ )  // modules 20 and 21 are fake...
// 	{
// 	  _pn[ilmod] = MEEEGeom::pn( ilmod );
// 	}
      //      abort();
    }

  TString t_name; 

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
  addBranchI( t_name, "SUB_RUN_START_LOW"   );
  addBranchI( t_name, "SUB_RUN_START_HIGH"  );
  addBranchI( t_name, "SUB_RUN_END_LOW"     );
  addBranchI( t_name, "SUB_RUN_END_HIGH"    );
  addBranchI( t_name, "DB_TIMESTAMP_LOW"    );
  addBranchI( t_name, "DB_TIMESTAMP_HIGH"   );
  addBranchC( t_name, "SUB_RUN_TYPE"   );

  if( _type==ME::iLaser )
    {  
      //
      // Laser ADC Primitives
      //
      bookHistoI( _primStr, "LOGIC_ID" );
      bookHistoI( _primStr, "FLAG" );
      bookHistoF( _primStr, "MEAN" );
      bookHistoF( _primStr, "RMS" );
      bookHistoF( _primStr, "M3" );
      bookHistoF( _primStr, "APD_OVER_PNA_MEAN" );
      bookHistoF( _primStr, "APD_OVER_PNA_RMS" );
      bookHistoF( _primStr, "APD_OVER_PNA_M3" );
      bookHistoF( _primStr, "APD_OVER_PNB_MEAN" );
      bookHistoF( _primStr, "APD_OVER_PNB_RMS" );
      bookHistoF( _primStr, "APD_OVER_PNB_M3" );
      bookHistoF( _primStr, "APD_OVER_PN_MEAN" );
      bookHistoF( _primStr, "APD_OVER_PN_RMS" );
      bookHistoF( _primStr, "APD_OVER_PN_M3" );
      bookHistoF( _primStr, "APD_OVER_APDA_MEAN" );
      bookHistoF( _primStr, "APD_OVER_APDA_RMS" );
      bookHistoF( _primStr, "APD_OVER_APDA_M3" );
      bookHistoF( _primStr, "APD_OVER_APDB_MEAN" );
      bookHistoF( _primStr, "APD_OVER_APDB_RMS" );
      bookHistoF( _primStr, "APD_OVER_APDB_M3" );
      bookHistoF( _primStr, "SHAPE_COR" );
      bookHistoF( _primStr, "ALPHA" );
      bookHistoF( _primStr, "BETA" );
      // NEW GHM 08/06 --> SCHEMA MODIFIED?
      bookHistoF( _primStr, "TIME_MEAN" ); 
      bookHistoF( _primStr, "TIME_RMS"  );
      bookHistoF( _primStr, "TIME_M3"   );  
      bookHistoF( _primStr, "TIME_NEVT" );

      //
      // Laser PN Primitives
      //
      t_name = lmfLaserName( ME::iLmfLaserPnPrim, _type, _color );
      addBranchI( t_name, "LOGIC_ID" );
      addBranchI( t_name, "FLAG"     );
      addBranchF( t_name, "MEAN"     );
      addBranchF( t_name, "RMS"      );
      addBranchF( t_name, "M3"     );
      addBranchF( t_name, "PNA_OVER_PNB_MEAN"     );
      addBranchF( t_name, "PNA_OVER_PNB_RMS"      );
      addBranchF( t_name, "PNA_OVER_PNB_M3"       );

      //
      // Laser Pulse
      //
      t_name = lmfLaserName( ME::iLmfLaserPulse, _type, _color );
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
      t_name = lmfLaserName( ME::iLmfLaserConfig, _type );
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
      // Laser LaserRun config dat
      //
      t_name = "RUN_LASERRUN_CONFIG_DAT";
      addBranchI( t_name, "LOGIC_ID"       );
      addBranchC( t_name, "LASER_SEQUENCE_TYPE"    );
      addBranchC( t_name, "LASER_SEQUENCE_COND"  );

    }
  else if( _type==ME::iTestPulse )
    {
      //
      // Test Pulse ADC Primitives
      //
      bookHistoI( _tpPrimStr, "LOGIC_ID" );
      bookHistoI( _tpPrimStr, "FLAG" );
      bookHistoF( _tpPrimStr, "MEAN" );
      bookHistoF( _tpPrimStr, "RMS" );
      bookHistoF( _tpPrimStr, "M3" );
      bookHistoF( _tpPrimStr, "NEVT" );

      //
      // Test Pulse PN Primitives
      //
      t_name = lmfLaserName( ME::iLmfTestPulsePnPrim, _type );
      addBranchI( t_name, "LOGIC_ID" );
      addBranchI( t_name, "FLAG"     );
      addBranchI( t_name, "GAIN"     );
      addBranchF( t_name, "MEAN"     );
      addBranchF( t_name, "RMS"      );
      addBranchF( t_name, "M3"     );

      //
      // Test Pulse Config
      //
      t_name = lmfLaserName( ME::iLmfTestPulseConfig, _type );
      addBranchI( t_name, "LOGIC_ID"       );
      addBranchI( t_name, "VFE_GAIN"       );
      addBranchI( t_name, "PN_GAIN"        );
    }
}

void
MELaserPrim::fillHistograms()
{
  TString t_name; 

  if( !init_ok ) return;

  Long64_t nb       = 0;
  Long64_t nentries = 0;
  Long64_t ientry   = 0;

  int channelView_(0);
  int id1_(0), id2_(0);
  int logic_id_(0);

  if( _type==ME::iLaser )
    {

      nentries = apdpn_tree->GetEntriesFast();
      for( Long64_t jentry=0; jentry<nentries; jentry++ ) 
	{      
	  ientry = apdpn_tree->LoadTree( jentry );
	  assert( ientry>=0 );
	  nb     = apdpn_tree->GetEntry( jentry );

	  if(ab_tree){
	    ientry = ab_tree->LoadTree( jentry );
	    assert( ientry>=0 );
	    nb     = ab_tree->GetEntry( jentry );
	  }


	  if( apdpn_iphi<0 ) continue;      

	  // fixme remove until coordinated are fine
	  //if(ab_tree) assert( apdpn_ieta==ab_ieta && apdpn_iphi==ab_iphi );
      
	  int ix(0);
	  int iy(0);
	  if( _isBarrel )
	    {
	      // Barrel, global coordinates
	      id1_ = _sm;   
	      if ( apdpn_side != _side ) continue; 
	      int ieta=apdpn_ieta;
	      int iphi=apdpn_iphi;
	      MEEBGeom::XYCoord xy_ = MEEBGeom::localCoord( ieta, iphi );
	      ix = xy_.first;
	      iy = xy_.second;
	      id2_ = MEEBGeom::crystal_channel( ix, iy ); 
	      channelView_ = iEB_crystal_number;
	    }
	  else
	    {
	      // EndCaps, global coordinates
	      id1_ = apdpn_iphi;
	      id2_ = apdpn_ieta; 
	      ix = id1_;
	      iy = id2_;
	      channelView_ = iEE_crystal_number;
	    }

	  logic_id_ = logicId( channelView_, id1_, id2_ );

	  int flag = apdpn_flag;

	  setInt( "LOGIC_ID",           ix, iy,  logic_id_ );
	  setInt( "FLAG",               ix, iy,  flag );
	  setVal( "MEAN",               ix, iy,  apdpn_apdpn[iAPD][iMean] );
	  setVal( "RMS",                ix, iy,  apdpn_apdpn[iAPD][iRMS] );
	  setVal( "M3",                 ix, iy,  apdpn_apdpn[iAPD][iM3] );  // fixme --- peak?
	  setVal( "APD_OVER_PNA_MEAN",  ix, iy,  apdpn_apdpn[iAPDoPNA][iMean] );
	  setVal( "APD_OVER_PNA_RMS",   ix, iy,  apdpn_apdpn[iAPDoPNA][iRMS] );
	  setVal( "APD_OVER_PNA_M3",    ix, iy,  apdpn_apdpn[iAPDoPNA][iM3] );  // fixme
	  setVal( "APD_OVER_PNB_MEAN",  ix, iy,  apdpn_apdpn[iAPDoPNB][iMean] );
	  setVal( "APD_OVER_PNB_RMS",   ix, iy,  apdpn_apdpn[iAPDoPNB][iRMS] );
	  setVal( "APD_OVER_PNB_M3",    ix, iy,  apdpn_apdpn[iAPDoPNB][iM3] );  // fixme
	  setVal( "APD_OVER_PN_MEAN",   ix, iy,  apdpn_apdpn[iAPDoPN][iMean] );
	  setVal( "APD_OVER_PN_RMS",    ix, iy,  apdpn_apdpn[iAPDoPN][iRMS] );
	  setVal( "APD_OVER_PN_M3",     ix, iy,  apdpn_apdpn[iAPDoPN][iM3] );  // fixme
	  // JM
	  setVal( "APD_OVER_APD_MEAN",  ix, iy,  apdpn_apdpn[iAPDoAPDA][iMean] );
	  setVal( "APD_OVER_APD_RMS",   ix, iy,  apdpn_apdpn[iAPDoAPDA][iRMS] );
	  setVal( "APD_OVER_APD_M3",    ix, iy,  apdpn_apdpn[iAPDoAPDA][iM3] );  // fixme
	  setVal( "APD_OVER_APDA_MEAN",  ix, iy,  apdpn_apdpn[iAPDoAPDA][iMean] );
	  setVal( "APD_OVER_APDA_RMS",   ix, iy,  apdpn_apdpn[iAPDoAPDA][iRMS] );
	  setVal( "APD_OVER_APDA_M3",    ix, iy,  apdpn_apdpn[iAPDoAPDA][iM3] );  // fixme
	  setVal( "APD_OVER_APDB_MEAN",  ix, iy,  apdpn_apdpn[iAPDoAPDB][iMean] );
	  setVal( "APD_OVER_APDB_RMS",   ix, iy,  apdpn_apdpn[iAPDoAPDB][iRMS] );
	  setVal( "APD_OVER_APDB_M3",    ix, iy,  apdpn_apdpn[iAPDoAPDB][iM3] );  // fixme
	  // JM
	  setVal( "SHAPE_COR",          ix, iy,  apdpn_ShapeCor );
	  if(ab_tree){
	    setVal( "ALPHA",              ix, iy,  ab_ab[iAlpha] );
	    setVal( "BETA",               ix, iy,  ab_ab[iBeta] );
	  }else{
	    setVal( "ALPHA",              ix, iy,  0. );
	    setVal( "BETA",               ix, iy,  0. );
	  }
	  // NEW GHM 08/06
	  setVal( "TIME_MEAN",          ix, iy,  apdpn_apdpn[iTime][iMean] );
	  setVal( "TIME_RMS",           ix, iy,  apdpn_apdpn[iTime][iRMS]  );
	  setVal( "TIME_M3",            ix, iy,  apdpn_apdpn[iTime][iM3]   );
	  setVal( "TIME_NEVT",          ix, iy,  apdpn_apdpn[iTime][iNevt] );

	}


      //
      // PN primitives
      //
      t_name = lmfLaserName( ME::iLmfLaserPnPrim, _type, _color );

      nentries = pn_tree->GetEntriesFast();
      assert( nentries%2==0 );
      int module_(0);
      id1_=_sm; id2_=0;
  
      Long64_t jentry=0;

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
	      
	      if( _side!=pn_side ) break;
		  
	      if( jj==1 ) assert( pn_moduleID==module_ );
	      module_ = pn_moduleID;
	      assert( pn_pnID==jj );
	      
	      // get the PN number
	      std::pair<int,int> memPn_ = ME::pn( _lmr, module_, (ME::PN)jj );
	      if( _isBarrel )
		{
		  //		  assert( memPn_.first%600==_dcc%600 );
		  id1_ = _sm;
		  id2_ = memPn_.second;
		}
	      else
		{
		  int dee_ = MEEEGeom::dee( _lmr );
		  //		  int mem_ = memPn_.first%600;
// 		  if(      dee_==1 )
// 		    {
// 		      if( jj==ME::iPNA ) assert( mem_==50 );
// 		      else               assert( mem_==51 );
// 		    }
// 		  else if( dee_==2 )
// 		    {
// 		      if( jj==ME::iPNA ) assert( mem_==47 );  // warning !
// 		      else               assert( mem_==46 );
// 		      //		      assert( mem_==46 || mem_==47 );
// 		    }
// 		  else if( dee_==3 )
// 		    {
// 		      if( jj==ME::iPNA ) assert( mem_==1 );
// 		      else               assert( mem_==2 );
// 		    }
// 		  else if( dee_==4 )
// 		    {
// 		      if( jj==ME::iPNA ) assert( mem_==5 );
// 		      else               assert( mem_==6 );
// 		    }
		  id1_ = dee_;
		  id2_ = (jj+1)*100+memPn_.second;
		}
	      
	      if( _isBarrel )
		{
		  channelView_ = iEB_LM_PN;
		}
	      else
		{
		  channelView_ = iEE_LM_PN;
		}
	      logic_id_ = logicId( channelView_, id1_, id2_ );
	      
	      i_t[t_name+separator+"LOGIC_ID"] = logic_id_;
	      f_t[t_name+separator+"MEAN"]  = pn_PN[iMean];
	      f_t[t_name+separator+"RMS"]   = pn_PN[iRMS];
	      f_t[t_name+separator+"M3"]  = pn_PN[iM3];    
	      f_t[t_name+separator+"PNA_OVER_PNB_MEAN"]  = (jj==0) ? pn_PNoPNB[iMean] : pn_PNoPNA[iMean];
	      f_t[t_name+separator+"PNA_OVER_PNB_RMS" ]  = (jj==0) ? pn_PNoPNB[iRMS]  : pn_PNoPNA[iRMS];
	      f_t[t_name+separator+"PNA_OVER_PNB_M3"]  = (jj==0) ? pn_PNoPNB[iM3] : pn_PNoPNA[iM3];
	      
	      t_t[t_name]->Fill();
	      
	    }
	  //      std::cout << "Module=" << module_ << "\tPNA=" << pn_[0] << "\tPNB=" << pn_[1] << std::endl;

// 	  if( _isBarrel )
// 	    jentry += 4;
// 	  else
// 	    jentry += 2;
	  jentry += 2;
	}

      logic_id_  = logicId( iECAL_LMR, _lmr );

      //
      // MATACQ primitives
      //

      if(mtq_tree){

	t_name = lmfLaserName( ME::iLmfLaserPulse, _type, _color );
    
	nentries = mtq_tree->GetEntriesFast();
	assert( nentries==2 );
	for( Long64_t jentry=0; jentry<nentries; jentry++ ) 
	  {        
	    ientry = mtq_tree->LoadTree( jentry );
	    assert( ientry>=0 );
	    nb     = mtq_tree->GetEntry( jentry );
	    
	    if ( mtq_side != _side ) continue; 
	    
	    i_t[t_name+separator+"LOGIC_ID"]       = logic_id_;
	    i_t[t_name+separator+"FIT_METHOD"]     = 0 ;   // fixme  --- what's this? ? ?
	    f_t[t_name+separator+"MTQ_AMPL"]       = mtq_mtq[iAmpl];
	    f_t[t_name+separator+"MTQ_TIME"]       = mtq_mtq[iPeak];
	    f_t[t_name+separator+"MTQ_RISE"]       = mtq_mtq[iTrise];
	    f_t[t_name+separator+"MTQ_FWHM"]       = mtq_mtq[iFwhm];
	    f_t[t_name+separator+"MTQ_FW20"]       = mtq_mtq[iFw20];
	    f_t[t_name+separator+"MTQ_FW80"]       = mtq_mtq[iFw80];
	    f_t[t_name+separator+"MTQ_SLIDING"]    = mtq_mtq[iSlide];      // fixme  --- sliding: max of average in sliding window
	
	    t_t[t_name]->Fill();
	  }
      }else{

	t_name = lmfLaserName( ME::iLmfLaserPulse, _type, _color );

	i_t[t_name+separator+"LOGIC_ID"]       = logic_id_;
	i_t[t_name+separator+"FIT_METHOD"]     =  0 ;   // fixme 
	f_t[t_name+separator+"MTQ_AMPL"]       = 0.0;
	f_t[t_name+separator+"MTQ_TIME"]       = 0.0;
	f_t[t_name+separator+"MTQ_RISE"]       = 0.0;
	f_t[t_name+separator+"MTQ_FWHM"]       = 0.0;
	f_t[t_name+separator+"MTQ_FW20"]       = 0.0;
	f_t[t_name+separator+"MTQ_FW80"]       = 0.0;
	f_t[t_name+separator+"MTQ_SLIDING"]    = 0.0;  

	t_t[t_name]->Fill();
      }

      //
      // Laser Run
      //
      t_name = lmfLaserName( ME::iLmfLaserRun, _type );
      //std::cout << "Fill "<< t_name << std::endl;
      i_t[t_name+separator+"LOGIC_ID"]       = logic_id_; 
      i_t[t_name+separator+"NEVENTS"]        =  _events;
      i_t[t_name+separator+"QUALITY_FLAG"]   = 1;                // fixme
      t_t[t_name]->Fill();
  
      //
      // Laser Config
      //
      t_name = lmfLaserName( ME::iLmfLaserConfig, _type );
      //std::cout << "Fill "<< t_name << std::endl;
      i_t[t_name+separator+"LOGIC_ID"]        = logic_id_;
      i_t[t_name+separator+"WAVELENGTH"]      = _color;
      i_t[t_name+separator+"VFE_GAIN"]        = _mgpagain; // fixme 
      i_t[t_name+separator+"PN_GAIN"]         = _memgain;  // fixme
      i_t[t_name+separator+"LSR_POWER"]       = _power; // will be available from MATACQ data
      i_t[t_name+separator+"LSR_ATTENUATOR"]  = _filter; // idem
      i_t[t_name+separator+"LSR_CURRENT"]     = 0; // idem
      i_t[t_name+separator+"LSR_DELAY_1"]     = _delay; // idem
      i_t[t_name+separator+"LSR_DELAY_2"]     = 0; // idem
      t_t[t_name]->Fill();

    }
  else if( _type==ME::iTestPulse )
    {

//       nentries = tpapd_tree->GetEntriesFast();
//       for( Long64_t jentry=0; jentry<nentries; jentry++ ) 
// 	{      
// 	  ientry = tpapd_tree->LoadTree( jentry );
// 	  assert( ientry>=0 );
// 	  nb     = tpapd_tree->GetEntry( jentry );

// 	  if( tpapd_iphi<0 ) continue;      

// 	  const bool new_= true;
      
// 	  int ix;
// 	  int iy;
// 	  int id2;
// 	  if( new_ )
// 	    {
// 	      // for Cruzet3 , global coordinates
// 	      if ( tpapd_side != _side ) continue; 
// 	      int ieta=tpapd_ieta;
// 	      int iphi=tpapd_iphi;
// 	      MEEBGeom::XYCoord xy_ = MEEBGeom::localCoord( ieta, iphi );
// 	      ix = xy_.first;
// 	      iy = xy_.second;
// 	      id2 = ix*20 + iy;  // !!! TO BE CHECKED !!!
// 	    }
// 	  else
// 	    {
// 	      // for Cruzet2 , local coordinates
// 	      ix = tpapd_ieta;
// 	      iy = 19-tpapd_iphi;
// 	      id2 = ix*20 + (20 - iy);  // !!! TO BE CHECKED !!!
// 	    }
// 	  // 

// 	  int id1 = _sm;   // fixme --- this is for barrel
// 	  int logic_id_ = 1011000000;    // fixme
// 	  logic_id_ += 10000*id1 + id2;

// 	  int flag = tpapd_flag;

// 	  setInt( "LOGIC_ID",           ix, iy,  logic_id_ );
// 	  setInt( "FLAG",               ix, iy,  flag );
// 	  setVal( "MEAN",               ix, iy,  tpapd_APD[iMean] );
// 	  setVal( "RMS",                ix, iy,  tpapd_APD[iRMS] );
// 	  setVal( "M3",                 ix, iy,  tpapd_APD[iM3] ); 
// 	  setVal( "NEVT",               ix, iy,  tpapd_APD[iNevt] );
// 	}

//       //
//       // PN primitives
//       //
//       t_name = lmfLaserName( ME::iLmfTestPulsePnPrim, _type );

//       nentries = tppn_tree->GetEntriesFast();
//       assert( nentries%2==0 );
//       int module_, pn_[2];
//       int id1_(_sm), id2_(0);
//       int logic_id_(0);
  
//       Long64_t jentry=0;
//       if( _side==1 ) jentry+=2;  // fixme : true also for endcaps?
//       while( jentry<nentries ) 
// 	{      
// 	  for( int jj=0; jj<2; jj++ )
// 	    {
// 	      // jj=0 --> PNA
// 	      // jj=1 --> PNB
	  
// 	      int zentry = jentry+jj;
// 	      assert( zentry<nentries );
	  
// 	      ientry = tppn_tree->LoadTree( zentry );
// 	      assert( ientry>=0 );
// 	      nb     = tppn_tree->GetEntry( zentry );

// 	      if( jj==1 ) assert( tppn_moduleID==module_ );
// 	      module_ = tppn_moduleID;
// 	      assert( tppn_pnID==jj );
	  
// 	      pn_[jj] = ( jj==0 ) ? _pn[module_].first : _pn[module_].second;
// 	      id2_ = pn_[jj];
// 	      logic_id_ = 1131000000 ;
// 	      //	  logic_id_ = 0;    // fixme
// 	      logic_id_ += 10000*id1_ + id2_;
	  
// 	      i_t[t_name+separator+"LOGIC_ID"] = logic_id_;
// 	      i_t[t_name+separator+"GAIN"]  = tppn_gain;
// 	      f_t[t_name+separator+"MEAN"]  = tppn_PN[iMean];
// 	      f_t[t_name+separator+"RMS"]   = tppn_PN[iRMS];
// 	      f_t[t_name+separator+"M3"]    = tppn_PN[iM3];     // fixme --- peak?

// 	      t_t[t_name]->Fill();

// 	    }

// 	  //      std::cout << "Module=" << module_ << "\tPNA=" << pn_[0] << "\tPNB=" << pn_[1] << std::endl;

// 	  jentry += 4;
// 	}

//       logic_id_ = 1041000000;
//       logic_id_ += 10000*id1_;
//       logic_id_ += id1_;
  
//       //
//       // Test Pulse Run
//       //
//       t_name = lmfLaserName( ME::iLmfTestPulseRun, _type );
//       //std::cout << "Fill "<< t_name << std::endl;
//       i_t[t_name+separator+"LOGIC_ID"]       = logic_id_;  // fixme --- is there a channelview for this?
//       i_t[t_name+separator+"NEVENTS"]        =  _events;
//       i_t[t_name+separator+"QUALITY_FLAG"]   = 1;                // fixme
//       t_t[t_name]->Fill();
  
//       //
//       // Test Pulse Config
//       //
//       t_name = lmfLaserName( ME::iLmfTestPulseConfig, _type );
//       //std::cout << "Fill "<< t_name << std::endl;
//       i_t[t_name+separator+"LOGIC_ID"]        = logic_id_;  // fixme
//       i_t[t_name+separator+"VFE_GAIN"]        = _mgpagain; // fixme 
//       i_t[t_name+separator+"PN_GAIN"]         = _memgain;  // fixme
//       t_t[t_name]->Fill();
    }

  //
  // Laser Run IOV
  //
  t_name = "LMF_RUN_IOV";
  //std::cout << "Fill "<< t_name << std::endl;
  i_t[t_name+separator+"TAG_ID"]        = 0;       // fixme
  i_t[t_name+separator+"SUB_RUN_NUM"]   = _run;      // fixme
  i_t[t_name+separator+"SUB_RUN_START_LOW" ] = ME::time_low( _ts_beg );
  i_t[t_name+separator+"SUB_RUN_START_HIGH"] = ME::time_high( _ts_beg );
  i_t[t_name+separator+"SUB_RUN_END_LOW"   ] = ME::time_low( _ts_end );
  i_t[t_name+separator+"SUB_RUN_END_HIGH"  ] = ME::time_high( _ts_end );
  i_t[t_name+separator+"DB_TIMESTAMP_LOW"  ] = ME::time_low( _ts ); 
  i_t[t_name+separator+"DB_TIMESTAMP_HIGH" ] = ME::time_high( _ts );
  c_t[t_name+separator+"SUB_RUN_TYPE"]  = "LASER TEST CRUZET"; //fixme
  t_t[t_name]->Fill();

}

void
MELaserPrim::writeHistograms()
{
  if( !init_ok ) return;

  out_file = new TFile( _outfile, "RECREATE" );
  //  out_file->cd();

  std::map< TString, TH2* >::iterator it;

  for( it=i_h.begin(); it!=i_h.end(); it++ )
    {
      it->second->Write();
      delete it->second;
    }

  for( it=f_h.begin(); it!=f_h.end(); it++ )
    {
      it->second->Write();
      delete it->second;
    }

  std::map< TString, TTree* >::iterator it_t;
  for( it_t=t_t.begin(); it_t!=t_t.end(); it_t++ )
    {
      it_t->second->Write();
      delete it_t->second;
    }

  //  std::cout << "Closing " << _outfile << std::endl;
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
  delete tpapd_tree;
  delete tppn_tree;
  if( apdpn_file!=0 )
    {
      //      std::cout << "Closing apdpn_file " << std::endl;
      apdpn_file->Close();
      delete apdpn_file;
      apdpn_file = 0;
    }
  if( ab_file!=0 )
    {
      //      std::cout << "Closing ab_file " << std::endl;
      ab_file->Close();
      delete ab_file;
      ab_file = 0;
    }
  if( mtq_file!=0 )
    {
      //      std::cout << "Closing mtq_file " << std::endl;
      mtq_file->Close();
      delete mtq_file;
      mtq_file = 0;
    }
  if( tpapd_file!=0 )
    {
      //      std::cout << "Closing tpapd_file " << std::endl;
      tpapd_file->Close();
      delete tpapd_file;
      tpapd_file = 0;
    }
}

void 
MELaserPrim::print( ostream& o )
{
  o << "DCC/SM/side/type/color/run/ts " << _dcc << "/" << _sm << "/" << _side << "/" 
    << _type << "/" << _color << "/" << _run << "/" << _ts << std::endl;
  
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
MELaserPrim::lmfLaserName( int table, int type, int color )
{
  TString str("LMF_ERROR");
  if( table<0 || table>=ME::iSizeLmf )  return str;
  if( color<0 || color>=ME::iSizeC )    return str;

  if( type==ME::iLaser )
    {
      TString colstr;
      switch( color )
	{
	case ME::iBlue:   colstr = "_BLUE"; break;
	case ME::iGreen:  colstr = "_GREEN"; break;
	case ME::iRed:    colstr = "_RED";  break;
	case ME::iIRed:   colstr = "_IRED";  break;
	default:  abort();
	}
      str = "LMF_LASER";
      switch( table )
	{
	case ME::iLmfLaserRun:      str  = "LMF_RUN";                 break; 
	case ME::iLmfLaserConfig:   str += "_CONFIG";                 break; 
	case ME::iLmfLaserPulse:    str += colstr; str += "_PULSE";   break; 
	case ME::iLmfLaserPrim:     str += colstr; str += "_PRIM";    break; 
	case ME::iLmfLaserPnPrim:   str += colstr; str += "_PN_PRIM"; break; 
	default:  abort();
	}
    }
  else if( type==ME::iTestPulse )
    {
      str = "LMF_TEST_PULSE";
      switch( table )
	{
	case ME::iLmfTestPulseRun:     str = "LMF_RUN";     break;
	case ME::iLmfTestPulseConfig:  str += "_CONFIG";    break;
	case ME::iLmfTestPulsePrim:    str += "_PRIM";      break;	  
	case ME::iLmfTestPulsePnPrim:  str += "_PN_PRIM";   break;
	default: abort();
	}
    }
  str += "_DAT";
  return str;
}

void
MELaserPrim::addBranchI( const char* t_name_, const char* v_name_ )
{
  TString slashI("/i"); // Warning: always unsigned
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
  TString name_;
  if( _type==ME::iLaser ) name_=_primStr+name;
  else if( _type==ME::iTestPulse ) name_=_tpPrimStr+name;
 
  int _ival = getInt( name_, ix, iy );
  assert( _ival!=-99 );
  if( _ival!=0 ) return false; 

  TH2I* h_ = (TH2I*) i_h[name_];
  assert( h_!=0 );
  h_->Fill( ix+0.5, iy+0.5, ival );

  return true;
}

bool    
MELaserPrim::setVal( const char* name, int ix, int iy, float val )
{
  TString name_;
  if( _type==ME::iLaser ) name_=_primStr+name;
  else if( _type==ME::iTestPulse ) name_=_tpPrimStr+name;
 
  float _val = getVal( name_, ix, iy );
  assert( _val!=-99 );
  if( _val!=0 ) return false; 

  TH2F* h_ = (TH2F*) f_h[name_];
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

  // ghm
  if( f_t.count(key_)!=1 )
    {
      std::cout << key_ << std::endl;
    }
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
  std::map< TString, TH2* >::iterator it;

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

  std::map< TString, TTree* >::iterator it_t;
  for( it_t=t_t.begin(); it_t!=t_t.end(); it_t++ )
    {
      delete it_t->second;
      it->second = 0;
    }
  t_t.clear();
}
