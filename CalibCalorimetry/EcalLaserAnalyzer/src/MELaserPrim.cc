#define MELaserPrim_cxx
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MELaserPrim.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEGeom.h"
#include <cassert>
#include <cstdlib>

TString MELaserPrim::apdpn_arrayName[MELaserPrim::iSizeArray_apdpn] = 
  {"APD", "APDoPN", "APDoPNA", "APDoPNB","APDoPNCor", "APDoPNACor", "APDoPNBCor",
   "APDoAPD", "APDoAPDA", "APDoAPDB", "Time"};
TString MELaserPrim::apdpnabfit_arrayName[MELaserPrim::iSizeArray_apdpnabfit] = 
  {"APDoPNCor", "APDoPNACor", "APDoPNBCor"};
TString MELaserPrim::apdpnabfix_arrayName[MELaserPrim::iSizeArray_apdpnabfix] = 
  {"APDoPNCor", "APDoPNACor", "APDoPNBCor"};

TString MELaserPrim::apdpn_varName[MELaserPrim::iSize_apdpn] = { "Mean", "RMS", "M3", "Nevt", "Min", "Max"};
TString MELaserPrim::apdpn_varUnit[MELaserPrim::iSizeArray_apdpn][MELaserPrim::iSize_apdpn] = 
 
  { { " (ADC Counts)", " (ADC Counts)", " (ADC Counts)" ,"", " (ADC Counts)", " (ADC Counts)"},
    {"", "", "", "", "", ""},
    {"", "", "", "", "", ""},
    {"", "", "", "", "", ""},
    {"", "", "", "", "", ""},
    {"", "", "", "", "", ""},
    {" (25 ns)", " (25 ns)", " (25 ns)", "", " (25 ns)", " (25 ns)"} };
TString MELaserPrim::apdpn_extraVarName[MELaserPrim::iSizeExtra_apdpn] = { "shapeCorAPD" };
TString MELaserPrim::apdpn_extraVarUnit[MELaserPrim::iSizeExtra_apdpn] = { "" };

TString MELaserPrim::ab_varName[MELaserPrim::iSize_ab] = { "alpha", "beta", "width", "chi2" };
TString MELaserPrim::mtq_varName[MELaserPrim::iSize_mtq] = {"peak", "sigma", "fit", "ampl", "trise", "fwhm", "fw20", "fw80", "sliding", "FWHM","FW10","FW05" };
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
  //  cout<<" entering MELaserPrim constructor"<< endl;// TESTJULIE

  apdpn_file =0;
  apdpnabfit_file =0;
  apdpnabfix_file =0;
  ab_file    =0;
  mtq_file   =0;
  tpapd_file =0;
  apdpn_tree =0;
  apdpnabfit_tree=0;
  apdpnabfix_tree=0;
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

  if(_dcc>600) _fed = _dcc;
  else _fed=_dcc+600;

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

  //  cout<<" type="<<_type<<" color="<< _color<< endl;// TESTJULIE
  //  cout<<" type="<<ME::type[_type]<<" color="<<ME::color[_color]<< endl;// TESTJULIE
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
  else if( _type==ME::iLED )
    {
      _LEDprimStr     = lmfLaserName( ME::iLmfLEDPrim,   _type, _color )+separator;
      _LEDpnPrimStr   = lmfLaserName( ME::iLmfLEDPnPrim, _type, _color )+separator; 
      //      cout<<" primstr "<<_LEDprimStr <<"  " <<_LEDpnPrimStr<< endl;// TESTJULIE
    }

  _lmr = ME::lmr( _dcc, _side );
  ME::regionAndSector( _lmr, _reg, _sm, _dcc, _side );    
  _isBarrel = (_reg==ME::iEBM || _reg==ME::iEBP);
  _sectorStr  = ME::smName( _lmr );
  _regionStr  = _sectorStr;
  _regionStr += "_"; _regionStr  += _side;

  //  cout<<" before init()"<< endl;// TESTJULIE
  init();
  //  cout<<" after init()"<< endl;// TESTJULIE
  bookHistograms();
  //  cout<<" after bookHistograms()"<< endl;// TESTJULIE
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

 
  bool verbose_ = true;

  if( _inpath=="0" )
    {
      if( verbose_ ) cout << "no input file" << endl;
      init_ok = true;
      return; // GHM
    }

  TString cur(_inpath);
  if( !cur.EndsWith("/") ) cur+="/";

  if( _type==ME::iLaser )
    {
      TString _APDPN_fname =cur; _APDPN_fname += "APDPN_LASER.root";
      TString _APDPNabfit_fname =cur; _APDPNabfit_fname += "APDPN_LASER_ABFit.root";
      TString _APDPNabfix_fname =cur; _APDPNabfix_fname += "APDPN_LASER_ABFix.root";
      TString _AB_fname    =cur; _AB_fname    += "AB";_AB_fname    += _fed;_AB_fname    += "_LASER.root"; 
      TString _MTQ_fname   =cur; _MTQ_fname   += "MATACQ.root";

      bool apdpn_ok, ab_ok, pn_ok, mtq_ok;
      bool apdpnabfit_ok,apdpnabfix_ok;
      apdpn_ok=false; ab_ok=false; pn_ok=false; mtq_ok=false;
      apdpnabfit_ok=false;apdpnabfix_ok=false;

      FILE *test; 
      test = fopen( _APDPN_fname,"r");  
      if (test) {
	apdpn_ok = true;
	pn_ok = true;
	fclose( test );
      }
      test = fopen( _APDPNabfit_fname,"r");  
      if (test) {
	apdpnabfit_ok = true;
	fclose( test );
      }
      test = fopen( _APDPNabfix_fname,"r");  
      if (test) {
	apdpnabfix_ok = true;
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

      if(apdpn_ok){
	apdpn_file = TFile::Open( _APDPN_fname ); 
	if (apdpn_file->IsZombie()) apdpn_ok=false;
      } 
      if(apdpnabfit_ok){
	apdpnabfit_file = TFile::Open( _APDPNabfit_fname );
	if (apdpnabfit_file->IsZombie()) apdpn_ok=false;
      }

      if(apdpnabfix_ok){
	apdpnabfix_file = TFile::Open( _APDPNabfix_fname );
	if (apdpnabfix_file->IsZombie()) apdpn_ok=false;
      }
      if(ab_ok){
	ab_file    = TFile::Open(    _AB_fname );
	if (ab_file->IsZombie()) apdpn_ok=false;
      }
      if(mtq_ok){
	mtq_file   = TFile::Open(   _MTQ_fname );
	if (mtq_file->IsZombie()) apdpn_ok=false;
      }
      
      if( verbose_ )
	{
	  cout << _APDPN_fname << " ok=" << apdpn_ok << endl;
	  cout << _AB_fname    << " ok=" << ab_ok    << endl;
	  cout << _MTQ_fname   << " ok=" << mtq_ok    << endl;
	}
      if (! apdpn_ok || ! pn_ok ) return; 
  
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
      if(apdpnabfit_ok)  {
	TTree *checkABFit = (TTree*) apdpnabfit_file->Get(apdpn_tree_name);
	if( checkABFit ==0 ) apdpnabfit_ok = false;
      } 
      if(apdpnabfix_ok)  {
	TTree *checkABFix = (TTree*) apdpnabfix_file->Get(apdpn_tree_name);
	if( checkABFix ==0 ) apdpnabfix_ok = false;
      }

      if(ab_ok)  {
	TTree *checkAB = (TTree*) ab_file->Get(ab_tree_name);
	if( checkAB ==0 ) ab_ok = false;
      }

      if( _color != ME::iIRed && _color != ME::iBlue ){ 
	cout << "MELaserPrim::init() -- Fatal Error -- Wrong Laser Color : " << _color << " ---- Abort " << endl;
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
      if( apdpn_tree->GetBranchStatus("shapeCorAPD")) apdpn_tree->SetBranchAddress("shapeCorAPD", &apdpn_shapeCorAPD, &b_apdpn_shapeCorAPD);
      else apdpn_shapeCorAPD = 0.0;
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
      ab_tree->SetBranchAddress("side",      &ab_side,      &b_ab_side      );
      for( int ii=0; ii<iSize_ab; ii++ )
	{
	  ab_tree->SetBranchAddress( ab_varName[ii], &ab_ab[ii], &b_ab_ab[ii] );
	}
      }

      pn_tree = (TTree*) apdpn_file->Get(pn_tree_name);
      assert( pn_tree!=0 );
      pn_tree->SetMakeClass(1);
      pn_tree->SetBranchAddress( "side",       &pn_side,       &b_pn_side       );
      pn_tree->SetBranchAddress( "pnID",       &pn_pnID,       &b_pn_pnID       );
      pn_tree->SetBranchAddress( "moduleID",   &pn_moduleID,   &b_pn_moduleID   );
      pn_tree->SetBranchAddress( "PN",          pn_PN,         &b_pn_PN         );
      pn_tree->SetBranchAddress( "PNoPN",       pn_PNoPN,      &b_pn_PNoPN      );
      pn_tree->SetBranchAddress( "PNoPNA",      pn_PNoPNA,     &b_pn_PNoPNA     );
      pn_tree->SetBranchAddress( "PNoPNB",      pn_PNoPNB,     &b_pn_PNoPNB     );
      pn_tree->SetBranchAddress( "shapeCorPN", &pn_shapeCorPN, &b_pn_shapeCorPN );

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

      if( apdpnabfit_ok ){
	apdpnabfit_tree = (TTree*) apdpnabfit_file->Get(apdpn_tree_name);
	apdpnabfit_file->SetName(apdpn_tree_name+"abfit");
	apdpnabfit_tree->SetMakeClass(1);
	apdpnabfit_tree->SetBranchAddress("side", &apdpnabfit_side, &b_apdpnabfit_side);
	apdpnabfit_tree->SetBranchAddress("ieta", &apdpnabfit_ieta, &b_apdpnabfit_ieta);
	apdpnabfit_tree->SetBranchAddress("iphi", &apdpnabfit_iphi, &b_apdpnabfit_iphi);
	apdpnabfit_tree->SetBranchAddress("flag", &apdpnabfit_flag, &b_apdpnabfit_flag);
	for( int jj=0; jj<iSizeArray_apdpnabfit; jj++ )
	  {
	    TString name_ = apdpnabfit_arrayName[jj];
	    apdpnabfit_tree->SetBranchAddress(name_, apdpn_apdpnabfit[jj], &b_apdpn_apdpnabfit[jj]);
	  }
      }

      if( apdpnabfix_ok ){
	apdpnabfix_tree = (TTree*) apdpnabfix_file->Get(apdpn_tree_name);
	apdpnabfix_file->SetName(apdpn_tree_name+"abfix");
	apdpnabfix_tree->SetMakeClass(1);
	apdpnabfix_tree->SetBranchAddress("side", &apdpnabfix_side, &b_apdpnabfix_side);
	apdpnabfix_tree->SetBranchAddress("ieta", &apdpnabfix_ieta, &b_apdpnabfix_ieta);
	apdpnabfix_tree->SetBranchAddress("iphi", &apdpnabfix_iphi, &b_apdpnabfix_iphi);
	apdpnabfix_tree->SetBranchAddress("flag", &apdpnabfix_flag, &b_apdpnabfix_flag);
	for( int jj=0; jj<iSizeArray_apdpnabfix; jj++ )
	  {
	    TString name_ = apdpnabfix_arrayName[jj];
	    apdpnabfix_tree->SetBranchAddress(name_, apdpn_apdpnabfix[jj], &b_apdpn_apdpnabfix[jj]);
	  }
      }
      
      

    }
  else if( _type==ME::iTestPulse )
    {
      TString _TPAPD_fname =cur; _TPAPD_fname += "APDPN_TESTPULSE.root";

      bool tpapd_ok;
      tpapd_ok=false; 
      bool tppn_ok;
      tppn_ok=false; 

      FILE *test; 
      test = fopen( _TPAPD_fname,"r");  
      if (test) {
	tpapd_ok = true;
	tppn_ok = true;
	fclose( test );
      }
	    

      if(tpapd_ok){
	tpapd_file = TFile::Open( _TPAPD_fname );
	if (tpapd_file->IsZombie() || !tpapd_file ){
	  tpapd_ok=false;
	  tppn_ok=false;
	}
      }
      if( verbose_ )
	{
	  cout << _TPAPD_fname << " apd ok=" << tpapd_ok << " pn ok=" << tppn_ok << endl;
	}
      if (!tpapd_ok || !tppn_ok ) return;
  
      TString tpapd_tree_name;
      TString tppn_tree_name;

      tpapd_tree_name = "TPAPD";
      tppn_tree_name  = "TPPN";   
      
      if(tpapd_ok)  {
	TTree *ckecktpapd= (TTree*) tpapd_file->Get(tpapd_tree_name);
	if( ckecktpapd ==0 ) tpapd_ok = false;
      }
      if(tppn_ok)  {
	TTree *ckecktppn= (TTree*) tpapd_file->Get(tppn_tree_name);
	if( ckecktppn ==0 ) tppn_ok = false;
      }
      if (!tpapd_ok || !tppn_ok ) return;
      
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
      tppn_tree->SetBranchAddress( "pnID",     &tppn_pnID,     &b_tppn_pnID     );
      tppn_tree->SetBranchAddress( "moduleID", &tppn_moduleID, &b_tppn_moduleID );
      tppn_tree->SetBranchAddress( "gain",     &tppn_gain,     &b_tppn_gain     );
      tppn_tree->SetBranchAddress( "PN",        tppn_PN,       &b_tppn_PN       );
    }
  else if( _type==ME::iLED )
    {
      TString _APDPN_fname =cur; _APDPN_fname += "APDPN_LED.root";
      
      bool apdpn_ok, pn_ok;
      apdpn_ok=false; pn_ok=false; 

      FILE *test; 
      test = fopen( _APDPN_fname,"r");  
      if (test) {
	apdpn_ok = true;
	pn_ok = true;
	fclose( test );
      }
      

      if(apdpn_ok){
	apdpn_file = TFile::Open( _APDPN_fname ); 
	if (apdpn_file->IsZombie()) apdpn_ok=false;
      } 
     
      if( verbose_ )
	{
	  cout << _APDPN_fname << " ok=" << apdpn_ok << endl;
	}
      if (! apdpn_ok || ! pn_ok ) return; 
  
      TString apdpn_tree_name;
      TString    pn_tree_name;

      apdpn_tree_name = "APDCol";
      pn_tree_name    = "PNCol";   
  
      apdpn_tree_name += _color;
      pn_tree_name    += _color;


      if( _color != ME::iRed && _color != ME::iBlue ){ 
	cout << "MELaserPrim::init() -- Fatal Error -- Wrong LED Color : " << _color << " ---- Abort " << endl;
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
      if( apdpn_tree->GetBranchStatus("shapeCorAPD")) apdpn_tree->SetBranchAddress("shapeCorAPD", &apdpn_shapeCorAPD, &b_apdpn_shapeCorAPD);
      else apdpn_shapeCorAPD = 0.0;
      for( int jj=0; jj<iSizeArray_apdpn; jj++ )
	{
	  TString name_ = apdpn_arrayName[jj];
	  apdpn_tree->SetBranchAddress(name_, apdpn_apdpn[jj], &b_apdpn_apdpn[jj]);
	}
  
      pn_tree = (TTree*) apdpn_file->Get(pn_tree_name);
      assert( pn_tree!=0 );
      pn_tree->SetMakeClass(1);
      pn_tree->SetBranchAddress( "side",       &pn_side,       &b_pn_side       );
      pn_tree->SetBranchAddress( "pnID",       &pn_pnID,       &b_pn_pnID       );
      pn_tree->SetBranchAddress( "moduleID",   &pn_moduleID,   &b_pn_moduleID   );
      pn_tree->SetBranchAddress( "PN",          pn_PN,         &b_pn_PN         );
      pn_tree->SetBranchAddress( "PNoPN",       pn_PNoPN,      &b_pn_PNoPN      );
      pn_tree->SetBranchAddress( "PNoPNA",      pn_PNoPNA,     &b_pn_PNoPNA     );
      pn_tree->SetBranchAddress( "PNoPNB",      pn_PNoPNB,     &b_pn_PNoPNB     );
      pn_tree->SetBranchAddress( "shapeCorPN", &pn_shapeCorPN, &b_pn_shapeCorPN );

     
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
      bookHistoF( _primStr, "NEVT" );
      bookHistoF( _primStr, "APD_OVER_PNA_MEAN" );
      bookHistoF( _primStr, "APD_OVER_PNA_RMS" );
      bookHistoF( _primStr, "APD_OVER_PNA_M3" );
      bookHistoF( _primStr, "APD_OVER_PNA_NEVT" );
      bookHistoF( _primStr, "APD_OVER_PNB_MEAN" );
      bookHistoF( _primStr, "APD_OVER_PNB_RMS" );
      bookHistoF( _primStr, "APD_OVER_PNB_M3" );
      bookHistoF( _primStr, "APD_OVER_PNB_NEVT" );
      bookHistoF( _primStr, "APD_OVER_PN_MEAN" );
      bookHistoF( _primStr, "APD_OVER_PN_RMS" );
      bookHistoF( _primStr, "APD_OVER_PN_M3" );
      bookHistoF( _primStr, "APD_OVER_PN_NEVT" );
      bookHistoF( _primStr, "APD_OVER_PNACOR_MEAN" );
      bookHistoF( _primStr, "APD_OVER_PNACOR_RMS" );
      bookHistoF( _primStr, "APD_OVER_PNACOR_M3" );
      bookHistoF( _primStr, "APD_OVER_PNACOR_NEVT" );
      bookHistoF( _primStr, "APD_OVER_PNBCOR_MEAN" );
      bookHistoF( _primStr, "APD_OVER_PNBCOR_RMS" );
      bookHistoF( _primStr, "APD_OVER_PNBCOR_M3" );
      bookHistoF( _primStr, "APD_OVER_PNBCOR_NEVT" );
      bookHistoF( _primStr, "APD_OVER_PNCOR_MEAN" );
      bookHistoF( _primStr, "APD_OVER_PNCOR_RMS" );
      bookHistoF( _primStr, "APD_OVER_PNCOR_M3" );
      bookHistoF( _primStr, "APD_OVER_PNCOR_NEVT" );
      bookHistoF( _primStr, "APD_OVER_APD_MEAN" );
      bookHistoF( _primStr, "APD_OVER_APD_RMS" );
      bookHistoF( _primStr, "APD_OVER_APD_M3" );
      bookHistoF( _primStr, "APD_OVER_APD_NEVT" );
      bookHistoF( _primStr, "APD_OVER_APDA_MEAN" );
      bookHistoF( _primStr, "APD_OVER_APDA_RMS" );
      bookHistoF( _primStr, "APD_OVER_APDA_M3" );
      bookHistoF( _primStr, "APD_OVER_APDA_NEVT" );
      bookHistoF( _primStr, "APD_OVER_APDB_MEAN" );
      bookHistoF( _primStr, "APD_OVER_APDB_RMS" );
      bookHistoF( _primStr, "APD_OVER_APDB_M3" );
      bookHistoF( _primStr, "APD_OVER_APDB_NEVT" );
      bookHistoF( _primStr, "SHAPE_COR_APD" );
      //    bookHistoF( _primStr, "ALPHA" );
      // bookHistoF( _primStr, "BETA" );
     //  bookHistoF( _primStr, "APD_OVER_PNACORABFIT_MEAN" );
//       bookHistoF( _primStr, "APD_OVER_PNACORABFIT_RMS" );
//       bookHistoF( _primStr, "APD_OVER_PNACORABFIT_M3" );
//       bookHistoF( _primStr, "APD_OVER_PNACORABFIT_NEVT" );
//       bookHistoF( _primStr, "APD_OVER_PNBCORABFIT_MEAN" );
//       bookHistoF( _primStr, "APD_OVER_PNBCORABFIT_RMS" );
//       bookHistoF( _primStr, "APD_OVER_PNBCORABFIT_M3" );
//       bookHistoF( _primStr, "APD_OVER_PNBCORABFIT_NEVT" );
//       bookHistoF( _primStr, "APD_OVER_PNCORABFIT_MEAN" );
//       bookHistoF( _primStr, "APD_OVER_PNCORABFIT_RMS" );
//       bookHistoF( _primStr, "APD_OVER_PNCORABFIT_M3" );
//       bookHistoF( _primStr, "APD_OVER_PNCORABFIT_NEVT" );
//       bookHistoF( _primStr, "APD_OVER_PNACORABFIX_MEAN" );
//       bookHistoF( _primStr, "APD_OVER_PNACORABFIX_RMS" );
//       bookHistoF( _primStr, "APD_OVER_PNACORABFIX_M3" );
//       bookHistoF( _primStr, "APD_OVER_PNACORABFIX_NEVT" );
//       bookHistoF( _primStr, "APD_OVER_PNBCORABFIX_MEAN" );
//       bookHistoF( _primStr, "APD_OVER_PNBCORABFIX_RMS" );
//       bookHistoF( _primStr, "APD_OVER_PNBCORABFIX_M3" );
//       bookHistoF( _primStr, "APD_OVER_PNBCORABFIX_NEVT" );
//       bookHistoF( _primStr, "APD_OVER_PNCORABFIX_MEAN" );
//       bookHistoF( _primStr, "APD_OVER_PNCORABFIX_RMS" );
//       bookHistoF( _primStr, "APD_OVER_PNCORABFIX_M3" );
//       bookHistoF( _primStr, "APD_OVER_PNCORABFIX_NEVT" );


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
      addBranchF( t_name, "NEVT"     );
      addBranchF( t_name, "PNA_OVER_PNB_MEAN"     );
      addBranchF( t_name, "PNA_OVER_PNB_RMS"      );
      addBranchF( t_name, "PNA_OVER_PNB_M3"       );
      addBranchF( t_name, "SHAPE_COR_PN"          );

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
      addBranchF( t_name, "MTQ_FW10"     );
      addBranchF( t_name, "MTQ_FW05"     );
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
  else if( _type==ME::iLED )
    {  
      //
      // Laser ADC Primitives
      //
      bookHistoI( _LEDprimStr, "LOGIC_ID" );
      bookHistoI( _LEDprimStr, "FLAG" );
      bookHistoF( _LEDprimStr, "MEAN" );
      bookHistoF( _LEDprimStr, "RMS" );
      bookHistoF( _LEDprimStr, "M3" );
      bookHistoF( _LEDprimStr, "NEVT" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNA_MEAN" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNA_RMS" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNA_M3" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNA_NEVT" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNB_MEAN" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNB_RMS" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNB_M3" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNB_NEVT" );
      bookHistoF( _LEDprimStr, "APD_OVER_PN_MEAN" );
      bookHistoF( _LEDprimStr, "APD_OVER_PN_RMS" );
      bookHistoF( _LEDprimStr, "APD_OVER_PN_M3" );
      bookHistoF( _LEDprimStr, "APD_OVER_PN_NEVT" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNACOR_MEAN" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNACOR_RMS" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNACOR_M3" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNACOR_NEVT" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNBCOR_MEAN" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNBCOR_RMS" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNBCOR_M3" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNBCOR_NEVT" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNCOR_MEAN" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNCOR_RMS" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNCOR_M3" );
      bookHistoF( _LEDprimStr, "APD_OVER_PNCOR_NEVT" );
      bookHistoF( _LEDprimStr, "APD_OVER_APD_MEAN" );
      bookHistoF( _LEDprimStr, "APD_OVER_APD_RMS" );
      bookHistoF( _LEDprimStr, "APD_OVER_APD_M3" );
      bookHistoF( _LEDprimStr, "APD_OVER_APD_NEVT" );
      bookHistoF( _LEDprimStr, "APD_OVER_APDA_MEAN" );
      bookHistoF( _LEDprimStr, "APD_OVER_APDA_RMS" );
      bookHistoF( _LEDprimStr, "APD_OVER_APDA_M3" );
      bookHistoF( _LEDprimStr, "APD_OVER_APDA_NEVT" );
      bookHistoF( _LEDprimStr, "APD_OVER_APDB_MEAN" );
      bookHistoF( _LEDprimStr, "APD_OVER_APDB_RMS" );
      bookHistoF( _LEDprimStr, "APD_OVER_APDB_M3" );
      bookHistoF( _LEDprimStr, "APD_OVER_APDB_NEVT" );
      bookHistoF( _LEDprimStr, "SHAPE_COR_APD" );
     
      // NEW GHM 08/06 --> SCHEMA MODIFIED?
      bookHistoF( _LEDprimStr, "TIME_MEAN" ); 
      bookHistoF( _LEDprimStr, "TIME_RMS"  );
      bookHistoF( _LEDprimStr, "TIME_M3"   );  
      bookHistoF( _LEDprimStr, "TIME_NEVT" );

      //
      // Laser PN Primitives
      //
      t_name = lmfLaserName( ME::iLmfLEDPnPrim, _type, _color );
      addBranchI( t_name, "LOGIC_ID" );
      addBranchI( t_name, "FLAG"     );
      addBranchF( t_name, "MEAN"     );
      addBranchF( t_name, "RMS"      );
      addBranchF( t_name, "M3"     );
      addBranchF( t_name, "NEVT"     );
      addBranchF( t_name, "PNA_OVER_PNB_MEAN"     );
      addBranchF( t_name, "PNA_OVER_PNB_RMS"      );
      addBranchF( t_name, "PNA_OVER_PNB_M3"       );
      addBranchF( t_name, "SHAPE_COR_PN"          );

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

      // Get APD/PN Results first
      //==========================

      nentries = apdpn_tree->GetEntriesFast();
      for( Long64_t jentry=0; jentry<nentries; jentry++ ) 
	{      
	  ientry = apdpn_tree->LoadTree( jentry );
	  assert( ientry>=0 );
	  nb     = apdpn_tree->GetEntry( jentry );


	  if( apdpn_iphi<0 ) continue;      

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
	      // EndCaps, global coordinates  //NEW CHANGED...
	      id1_ = apdpn_ieta;
	      id2_ = apdpn_iphi; 
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
	  setVal( "M3",                 ix, iy,  apdpn_apdpn[iAPD][iM3] );  
	  setVal( "NEVT",               ix, iy,  apdpn_apdpn[iAPD][iNevt] );  
	  setVal( "APD_OVER_PNA_MEAN",  ix, iy,  apdpn_apdpn[iAPDoPNA][iMean] );
	  setVal( "APD_OVER_PNA_RMS",   ix, iy,  apdpn_apdpn[iAPDoPNA][iRMS] );
	  setVal( "APD_OVER_PNA_M3",    ix, iy,  apdpn_apdpn[iAPDoPNA][iM3] );
	  setVal( "APD_OVER_PNA_NEVT",  ix, iy,  apdpn_apdpn[iAPDoPNA][iNevt] ); 
	  setVal( "APD_OVER_PNB_MEAN",  ix, iy,  apdpn_apdpn[iAPDoPNB][iMean] );
	  setVal( "APD_OVER_PNB_RMS",   ix, iy,  apdpn_apdpn[iAPDoPNB][iRMS] );
	  setVal( "APD_OVER_PNB_M3",    ix, iy,  apdpn_apdpn[iAPDoPNB][iM3] );  
	  setVal( "APD_OVER_PNB_NEVT",  ix, iy,  apdpn_apdpn[iAPDoPNB][iNevt] );
	  setVal( "APD_OVER_PN_MEAN",   ix, iy,  apdpn_apdpn[iAPDoPN][iMean] );
	  setVal( "APD_OVER_PN_RMS",    ix, iy,  apdpn_apdpn[iAPDoPN][iRMS] );
	  setVal( "APD_OVER_PN_M3",     ix, iy,  apdpn_apdpn[iAPDoPN][iM3] );  
	  setVal( "APD_OVER_PN_NEVT",   ix, iy,  apdpn_apdpn[iAPDoPN][iNevt] );
	 
	  setVal( "APD_OVER_PNACOR_MEAN",  ix, iy,  apdpn_apdpn[iAPDoPNACor][iMean] );
	  setVal( "APD_OVER_PNACOR_RMS",   ix, iy,  apdpn_apdpn[iAPDoPNACor][iRMS] );
	  setVal( "APD_OVER_PNACOR_M3",    ix, iy,  apdpn_apdpn[iAPDoPNACor][iM3] );  
	  setVal( "APD_OVER_PNACOR_NEVT",  ix, iy,  apdpn_apdpn[iAPDoPNACor][iNevt] );
	  setVal( "APD_OVER_PNBCOR_MEAN",  ix, iy,  apdpn_apdpn[iAPDoPNBCor][iMean] );
	  setVal( "APD_OVER_PNBCOR_RMS",   ix, iy,  apdpn_apdpn[iAPDoPNBCor][iRMS] );
	  setVal( "APD_OVER_PNBCOR_M3",    ix, iy,  apdpn_apdpn[iAPDoPNBCor][iM3] );  
	  setVal( "APD_OVER_PNBCOR_NEVT",  ix, iy,  apdpn_apdpn[iAPDoPNBCor][iNevt] );
	  setVal( "APD_OVER_PNCOR_MEAN",   ix, iy,  apdpn_apdpn[iAPDoPNCor][iMean] );
	  setVal( "APD_OVER_PNCOR_RMS",    ix, iy,  apdpn_apdpn[iAPDoPNCor][iRMS] );
	  setVal( "APD_OVER_PNCOR_M3",     ix, iy,  apdpn_apdpn[iAPDoPNCor][iM3] );  
	  setVal( "APD_OVER_PNCOR_NEVT",   ix, iy,  apdpn_apdpn[iAPDoPNCor][iNevt] );

	  setVal( "APD_OVER_APD_MEAN", ix, iy,  apdpn_apdpn[iAPDoAPD][iMean] );
	  setVal( "APD_OVER_APD_RMS",  ix, iy,  apdpn_apdpn[iAPDoAPD][iRMS] );
	  setVal( "APD_OVER_APD_M3",   ix, iy,  apdpn_apdpn[iAPDoAPD][iM3] );  
	  setVal( "APD_OVER_APD_NEVT",  ix, iy,  apdpn_apdpn[iAPDoAPD][iNevt] );
	  setVal( "APD_OVER_APDA_MEAN", ix, iy,  apdpn_apdpn[iAPDoAPDA][iMean] );
	  setVal( "APD_OVER_APDA_RMS",  ix, iy,  apdpn_apdpn[iAPDoAPDA][iRMS] );
	  setVal( "APD_OVER_APDA_M3",   ix, iy,  apdpn_apdpn[iAPDoAPDA][iM3] ); 
	  setVal( "APD_OVER_APDA_NEVT",  ix, iy,  apdpn_apdpn[iAPDoAPDA][iNevt] );
	  setVal( "APD_OVER_APDB_MEAN", ix, iy,  apdpn_apdpn[iAPDoAPDB][iMean] );
	  setVal( "APD_OVER_APDB_RMS",   ix, iy,  apdpn_apdpn[iAPDoAPDB][iRMS] );
	  setVal( "APD_OVER_APDB_M3",    ix, iy,  apdpn_apdpn[iAPDoAPDB][iM3] );  
	  setVal( "APD_OVER_APDB_NEVT",  ix, iy,  apdpn_apdpn[iAPDoAPDB][iNevt] );
	  // JM
	  setVal( "SHAPE_COR_APD",          ix, iy,  apdpn_shapeCorAPD );
	
	  // NEW GHM 08/06
	  setVal( "TIME_MEAN",          ix, iy,  apdpn_apdpn[iTime][iMean] );
	  setVal( "TIME_RMS",           ix, iy,  apdpn_apdpn[iTime][iRMS]  );
	  setVal( "TIME_M3",            ix, iy,  apdpn_apdpn[iTime][iM3]   );
	  setVal( "TIME_NEVT",          ix, iy,  apdpn_apdpn[iTime][iNevt] );

	}
    
      // Get AB Results:
      //================

      if( ab_tree ){
	nentries = ab_tree->GetEntriesFast();
	for( Long64_t jentry=0; jentry<nentries; jentry++ ) 
	  {      
	    ientry = ab_tree->LoadTree( jentry );
	    assert( ientry>=0 );
	    nb     = ab_tree->GetEntry( jentry );
	    
	    if( ab_iphi<0 ) continue;      
	    
	    int ix(0);
	    int iy(0);
	    if( _isBarrel )
	      {
		// Barrel, global coordinates
		id1_ = _sm;   
		if ( ab_side != _side ) continue; 
		int ieta=ab_ieta;
		int iphi=ab_iphi;
		MEEBGeom::XYCoord xy_ = MEEBGeom::localCoord( ieta, iphi );
		ix = xy_.first;
		iy = xy_.second;
		id2_ = MEEBGeom::crystal_channel( ix, iy ); 
		channelView_ = iEB_crystal_number;
	      }
	    else
	      {
		// EndCaps, global coordinates  //NEW CHANGED...
		id1_ = ab_ieta;
		id2_ = ab_iphi; 
		ix = id1_;
		iy = id2_;
		channelView_ = iEE_crystal_number;
	      }
	    
	    logic_id_ = logicId( channelView_, id1_, id2_ );
	    
	    //int flag = ab_flag;
	    setVal( "ALPHA",              ix, iy,  ab_ab[iAlpha] );
	    setVal( "BETA",               ix, iy,  ab_ab[iBeta] );
	    
	  }
      }
      
      // Get APD/PN fit Results
      //=============================

	if(apdpnabfit_tree){
	  nentries = apdpnabfit_tree->GetEntriesFast();
	  for( Long64_t jentry=0; jentry<nentries; jentry++ ) 
	    {      
	      ientry = apdpnabfit_tree->LoadTree( jentry );
	      assert( ientry>=0 );
	      nb     = apdpnabfit_tree->GetEntry( jentry );
	      

	  if( apdpnabfit_iphi<0 ) continue;      

      
	  int ix(0);
	  int iy(0);
	  if( _isBarrel )
	    {
	      // Barrel, global coordinates
	      id1_ = _sm;   
	      if ( apdpnabfit_side != _side ) continue; 
	      int ieta=apdpnabfit_ieta;
	      int iphi=apdpnabfit_iphi;
	      MEEBGeom::XYCoord xy_ = MEEBGeom::localCoord( ieta, iphi );
	      ix = xy_.first;
	      iy = xy_.second;
	      id2_ = MEEBGeom::crystal_channel( ix, iy ); 
	      channelView_ = iEB_crystal_number;
	    }
	  else
	    {
	      // EndCaps, global coordinates  //NEW CHANGED...
	      id1_ = apdpnabfit_ieta;
	      id2_ = apdpnabfit_iphi; 
	      ix = id1_;
	      iy = id2_;
	      channelView_ = iEE_crystal_number;
	    }
	  
	  //int flag = apdpnabfit_flag;
	  
// 	  setVal( "APD_OVER_PNACORABFIT_MEAN",  ix, iy,  apdpn_apdpnabfit[iAPDoPNACorabfit][iMean] );
// 	  setVal( "APD_OVER_PNACORABFIT_RMS",   ix, iy,  apdpn_apdpnabfit[iAPDoPNACorabfit][iRMS] );
// 	  setVal( "APD_OVER_PNACORABFIT_M3",    ix, iy,  apdpn_apdpnabfit[iAPDoPNACorabfit][iM3] );  
// 	  setVal( "APD_OVER_PNACORABFIT_NEVT",  ix, iy,  apdpn_apdpnabfit[iAPDoPNACorabfit][iNevt] );
// 	  setVal( "APD_OVER_PNBCORABFIT_MEAN",  ix, iy,  apdpn_apdpnabfit[iAPDoPNBCorabfit][iMean] );
// 	  setVal( "APD_OVER_PNBCORABFIT_RMS",   ix, iy,  apdpn_apdpnabfit[iAPDoPNBCorabfit][iRMS] );
// 	  setVal( "APD_OVER_PNBCORABFIT_M3",    ix, iy,  apdpn_apdpnabfit[iAPDoPNBCorabfit][iM3] );  
// 	  setVal( "APD_OVER_PNBCORABFIT_NEVT",  ix, iy,  apdpn_apdpnabfit[iAPDoPNBCorabfit][iNevt] );
// 	  setVal( "APD_OVER_PNCORABFIT_MEAN",   ix, iy,  apdpn_apdpnabfit[iAPDoPNCorabfit][iMean] );
// 	  setVal( "APD_OVER_PNCORABFIT_RMS",    ix, iy,  apdpn_apdpnabfit[iAPDoPNCorabfit][iRMS] );
// 	  setVal( "APD_OVER_PNCORABFIT_M3",     ix, iy,  apdpn_apdpnabfit[iAPDoPNCorabfit][iM3] );  
// 	  setVal( "APD_OVER_PNCORABFIT_NEVT",   ix, iy,  apdpn_apdpnabfit[iAPDoPNCorabfit][iNevt] );
	  
	    }
	}


      // Get APD/PN fix Results
      //=============================

	if(apdpnabfix_tree){
	  nentries = apdpnabfix_tree->GetEntriesFast();
	  for( Long64_t jentry=0; jentry<nentries; jentry++ ) 
	    {      
	      ientry = apdpnabfix_tree->LoadTree( jentry );
	      assert( ientry>=0 );
	      nb     = apdpnabfix_tree->GetEntry( jentry );
	      

	  if( apdpnabfix_iphi<0 ) continue;      

      
	  int ix(0);
	  int iy(0);
	  if( _isBarrel )
	    {
	      // Barrel, global coordinates
	      id1_ = _sm;   
	      if ( apdpnabfix_side != _side ) continue; 
	      int ieta=apdpnabfix_ieta;
	      int iphi=apdpnabfix_iphi;
	      MEEBGeom::XYCoord xy_ = MEEBGeom::localCoord( ieta, iphi );
	      ix = xy_.first;
	      iy = xy_.second;
	      id2_ = MEEBGeom::crystal_channel( ix, iy ); 
	      channelView_ = iEB_crystal_number;
	    }
	  else
	    {
	      // EndCaps, global coordinates  //NEW CHANGED...
	      id1_ = apdpnabfix_ieta;
	      id2_ = apdpnabfix_iphi; 
	      ix = id1_;
	      iy = id2_;
	      channelView_ = iEE_crystal_number;
	    }
	  
	  //int flag = apdpnabfix_flag;
	  
	 //  setVal( "APD_OVER_PNACORABFIX_MEAN",  ix, iy,  apdpn_apdpnabfix[iAPDoPNACorabfix][iMean] );
// 	  setVal( "APD_OVER_PNACORABFIX_RMS",   ix, iy,  apdpn_apdpnabfix[iAPDoPNACorabfix][iRMS] );
// 	  setVal( "APD_OVER_PNACORABFIX_M3",    ix, iy,  apdpn_apdpnabfix[iAPDoPNACorabfix][iM3] );  
// 	  setVal( "APD_OVER_PNACORABFIX_NEVT",  ix, iy,  apdpn_apdpnabfix[iAPDoPNACorabfix][iNevt] );
// 	  setVal( "APD_OVER_PNBCORABFIX_MEAN",  ix, iy,  apdpn_apdpnabfix[iAPDoPNBCorabfix][iMean] );
// 	  setVal( "APD_OVER_PNBCORABFIX_RMS",   ix, iy,  apdpn_apdpnabfix[iAPDoPNBCorabfix][iRMS] );
// 	  setVal( "APD_OVER_PNBCORABFIX_M3",    ix, iy,  apdpn_apdpnabfix[iAPDoPNBCorabfix][iM3] );  
// 	  setVal( "APD_OVER_PNBCORABFIX_NEVT",  ix, iy,  apdpn_apdpnabfix[iAPDoPNBCorabfix][iNevt] );
// 	  setVal( "APD_OVER_PNCORABFIX_MEAN",   ix, iy,  apdpn_apdpnabfix[iAPDoPNCorabfix][iMean] );
// 	  setVal( "APD_OVER_PNCORABFIX_RMS",    ix, iy,  apdpn_apdpnabfix[iAPDoPNCorabfix][iRMS] );
// 	  setVal( "APD_OVER_PNCORABFIX_M3",     ix, iy,  apdpn_apdpnabfix[iAPDoPNCorabfix][iM3] );  
// 	  setVal( "APD_OVER_PNCORABFIX_NEVT",   ix, iy,  apdpn_apdpnabfix[iAPDoPNCorabfix][iNevt] );
	  
	    }
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
		  
	      //if( jj==1 ) assert( pn_moduleID==module_ ); what is this for?
	      module_ = pn_moduleID;
	      assert( pn_pnID==jj );
	      
	      // get the PN number
	      pair<int,int> memPn_ = ME::pn( _lmr, module_, (ME::PN)jj );
	      if( _isBarrel )
		{
		  id1_ = _sm;
		  id2_ = memPn_.second;
		}
	      else
		{
		  int dee_ = MEEEGeom::dee( _lmr );

		  id1_ = dee_;
		  id2_ = (jj+1)*100+memPn_.second;
		}
	      
	      //cout <<"  pn prim " <<id1_<<"  " <<id2_<<"  " <<_lmr<<"  " <<module_<<"  " << jj<< endl;

	      if( _isBarrel )
		{
		  channelView_ = iEB_LM_PN;
		}
	      else
		{
		  channelView_ = iEE_LM_PN;
		}
	      logic_id_ = logicId( channelView_, id1_, id2_ );
	      int pnflag=0;
	      if(pn_PN[iMean]>200. &&pn_PN[iMean]<5000.) pnflag=1;
	      i_t[t_name+separator+"LOGIC_ID"] = logic_id_;
	      f_t[t_name+separator+"MEAN"]  = pn_PN[iMean];
	      i_t[t_name+separator+"FLAG"]  = pnflag;
	      f_t[t_name+separator+"RMS"]   = pn_PN[iRMS];
	      f_t[t_name+separator+"M3"]  = pn_PN[iM3];   
	      f_t[t_name+separator+"NEVT"]  = pn_PN[iNevt];    
	      f_t[t_name+separator+"PNA_OVER_PNB_MEAN"]  = (jj==0) ? pn_PNoPNB[iMean] : pn_PNoPNA[iMean];
	      f_t[t_name+separator+"PNA_OVER_PNB_RMS" ]  = (jj==0) ? pn_PNoPNB[iRMS]  : pn_PNoPNA[iRMS];
	      f_t[t_name+separator+"PNA_OVER_PNB_M3"]  = (jj==0) ? pn_PNoPNB[iM3] : pn_PNoPNA[iM3];
	      f_t[t_name+separator+"SHAPE_COR_PN"]  = pn_shapeCorPN;   

	      
	      t_t[t_name]->Fill();
	      
	    }
	  //      cout << "Module=" << module_ << "\tPNA=" << pn_[0] << "\tPNB=" << pn_[1] << endl;
	  
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
	assert( nentries<=2 );
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
	    f_t[t_name+separator+"MTQ_FWHM"]       = mtq_mtq[iFWHM];
	    f_t[t_name+separator+"MTQ_FW10"]       = mtq_mtq[iFW10];
	    f_t[t_name+separator+"MTQ_FW05"]       = mtq_mtq[iFW05];
	    f_t[t_name+separator+"MTQ_SLIDING"]    = mtq_mtq[iSlide]; 	
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
	f_t[t_name+separator+"MTQ_FW10"]       = 0.0;
	f_t[t_name+separator+"MTQ_FW05"]       = 0.0;
	f_t[t_name+separator+"MTQ_SLIDING"]    = 0.0;  

	t_t[t_name]->Fill();
      }

      //
      // Laser Run
      //
      t_name = lmfLaserName( ME::iLmfLaserRun, _type );
      //cout << "Fill "<< t_name << endl;
      i_t[t_name+separator+"LOGIC_ID"]       = logic_id_; 
      i_t[t_name+separator+"NEVENTS"]        =  _events;
      i_t[t_name+separator+"QUALITY_FLAG"]   = 1;                // fixme
      t_t[t_name]->Fill();
  
      //
      // Laser Config
      //
      t_name = lmfLaserName( ME::iLmfLaserConfig, _type );
      //cout << "Fill "<< t_name << endl;
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

      nentries = tpapd_tree->GetEntriesFast();
      for( Long64_t jentry=0; jentry<nentries; jentry++ ) 
	{      
	  ientry = tpapd_tree->LoadTree( jentry );
	  assert( ientry>=0 );
	  nb     = tpapd_tree->GetEntry( jentry );

	  if( tpapd_iphi<0 ) continue;      

	  int ix;
	  int iy;

	  if( _isBarrel )
	    {
	      // Barrel, global coordinates
	      id1_ = _sm;   
	      if ( tpapd_side  != _side ) continue; 
	      int ieta=tpapd_ieta;
	      int iphi=tpapd_iphi;
	      MEEBGeom::XYCoord xy_ = MEEBGeom::localCoord( ieta, iphi );
	      ix = xy_.first;
	      iy = xy_.second;
	      id2_ = MEEBGeom::crystal_channel( ix, iy ); 
	      channelView_ = iEB_crystal_number;
	    }
	  else
	    {
	      // EndCaps, global coordinates
	      id1_ = tpapd_ieta;
	      id2_ = tpapd_iphi; 
	      ix = id1_;
	      iy = id2_;
	      channelView_ = iEE_crystal_number;
	    }

	  int logic_id_ = logicId( channelView_, id1_, id2_ );
	  //	  cout << "ok coord "<< ix<<" "<<iy<<" "<<logic_id_<<" "<< jentry<<" "<< nentries<<endl;

	  int flag = tpapd_flag;

	  setInt( "LOGIC_ID",           ix, iy,  logic_id_ );
	  setInt( "FLAG",               ix, iy,  flag );
	  setVal( "MEAN",               ix, iy,  tpapd_APD[iMean] );
	  setVal( "RMS",                ix, iy,  tpapd_APD[iRMS] );
	  setVal( "M3",                 ix, iy,  tpapd_APD[iM3] ); 
	  setVal( "NEVT",               ix, iy,  tpapd_APD[iNevt] );
	  
	}

      //
      // PN primitives
      //
      
      t_name = lmfLaserName( ME::iLmfTestPulsePnPrim, _type );

      nentries = tppn_tree->GetEntriesFast();
      assert( nentries%2==0 );
      int module_(0);
      int id1_(_sm), id2_(0);
      int logic_id_(0);
  
      Long64_t jentry=0;
      
      if( _side==1 && !_isBarrel) jentry+=4; 
      
      while( jentry<nentries ) 
	{      
	  for( int jj=0; jj<2; jj++ )
	    {
	      // jj=0 --> PNA
	      // jj=1 --> PNB
	      
	      int zentry = jentry+jj;
	      //	      cout << "ok jentry "<< jentry<<"  zentry " <<zentry<< " nentries " <<nentries<< endl;
	      assert( zentry<nentries );
	      
	      ientry = tppn_tree->LoadTree( zentry );
	      assert( ientry>=0 );
	      nb     = tppn_tree->GetEntry( zentry );
	      
	      // tppn_side not in tree !!!
	      if( _isBarrel ){
		if(tppn_moduleID%2==0) tppn_side=1;
		else tppn_side=0;
	      }else{
		tppn_side=_side;
	      }
	      
	      //	      cout << "ok side "<< tppn_side <<" "<< tppn_moduleID<<" "<< _side<<endl;
	      
	      if( _side!=tppn_side ) break;
		  
	      //if( jj==1 ) assert( tppn_moduleID==module_ ); // WHAT???????
	      module_ = tppn_moduleID;
	      assert( tppn_pnID==jj );
	      
	      // get the PN number
	      pair<int,int> memPn_ = ME::pn( _lmr, module_, (ME::PN)jj );
	      if( _isBarrel )
		{
		  id1_ = _sm;
		  id2_ = memPn_.second;
		}
	      else
		{
		  int dee_ = MEEEGeom::dee( _lmr );
		  
		  id1_ = dee_;
		  id2_ = (jj+1)*100+memPn_.second;
		}
	      
	      //	      cout << "ok pn "<<id1_<<" "<<id2_<< endl;
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
	      i_t[t_name+separator+"GAIN"]  = tppn_gain;
	      f_t[t_name+separator+"MEAN"]  = tppn_PN[iMean];
	      f_t[t_name+separator+"RMS"]   = tppn_PN[iRMS];
	      f_t[t_name+separator+"M3"]    = tppn_PN[iM3]; 

	      //	      cout << "ok ampl "<<tppn_PN[iMean]<< endl;
	      t_t[t_name]->Fill();

	    }
	  jentry += 2;
	}

  
      //
      // Test Pulse Run
      //

      t_name = lmfLaserName( ME::iLmfTestPulseRun, _type );

      // fixme --- is there a channelview for this?
      logic_id_  = logicId( iECAL_LMR, _lmr );
      i_t[t_name+separator+"LOGIC_ID"]       = logic_id_; 
      
      i_t[t_name+separator+"NEVENTS"]        =  _events;
      i_t[t_name+separator+"QUALITY_FLAG"]   = 1;                // fixme
      t_t[t_name]->Fill();
    
      //
      // Test Pulse Config
      //
      t_name = lmfLaserName( ME::iLmfTestPulseConfig, _type );
      //cout << "Fill "<< t_name << endl;
      i_t[t_name+separator+"LOGIC_ID"]        = logic_id_;  // fixme
      i_t[t_name+separator+"VFE_GAIN"]        = _mgpagain; // fixme 
      i_t[t_name+separator+"PN_GAIN"]         = _memgain;  // fixme
      t_t[t_name]->Fill();
    }
  else if( _type==ME::iLED )
    {
      
      //      cout<< "Inside LED fill histograms "<< endl ; //TESTJULIE

      // Get APD/PN Results first
      //==========================

      nentries = apdpn_tree->GetEntriesFast();
      for( Long64_t jentry=0; jentry<nentries; jentry++ ) 
	{      
	  ientry = apdpn_tree->LoadTree( jentry );
	  assert( ientry>=0 );
	  nb     = apdpn_tree->GetEntry( jentry );


	  if( apdpn_iphi<0 ) continue;      

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
	      // EndCaps, global coordinates  //NEW CHANGED...
	      id1_ = apdpn_ieta;
	      id2_ = apdpn_iphi; 
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
	  setVal( "M3",                 ix, iy,  apdpn_apdpn[iAPD][iM3] );  
	  setVal( "NEVT",               ix, iy,  apdpn_apdpn[iAPD][iNevt] );  
	  setVal( "APD_OVER_PNA_MEAN",  ix, iy,  apdpn_apdpn[iAPDoPNA][iMean] );
	  setVal( "APD_OVER_PNA_RMS",   ix, iy,  apdpn_apdpn[iAPDoPNA][iRMS] );
	  setVal( "APD_OVER_PNA_M3",    ix, iy,  apdpn_apdpn[iAPDoPNA][iM3] );
	  setVal( "APD_OVER_PNA_NEVT",  ix, iy,  apdpn_apdpn[iAPDoPNA][iNevt] ); 
	  setVal( "APD_OVER_PNB_MEAN",  ix, iy,  apdpn_apdpn[iAPDoPNB][iMean] );
	  setVal( "APD_OVER_PNB_RMS",   ix, iy,  apdpn_apdpn[iAPDoPNB][iRMS] );
	  setVal( "APD_OVER_PNB_M3",    ix, iy,  apdpn_apdpn[iAPDoPNB][iM3] );  
	  setVal( "APD_OVER_PNB_NEVT",  ix, iy,  apdpn_apdpn[iAPDoPNB][iNevt] );
	  setVal( "APD_OVER_PN_MEAN",   ix, iy,  apdpn_apdpn[iAPDoPN][iMean] );
	  setVal( "APD_OVER_PN_RMS",    ix, iy,  apdpn_apdpn[iAPDoPN][iRMS] );
	  setVal( "APD_OVER_PN_M3",     ix, iy,  apdpn_apdpn[iAPDoPN][iM3] );  
	  setVal( "APD_OVER_PN_NEVT",   ix, iy,  apdpn_apdpn[iAPDoPN][iNevt] );
	 
	  setVal( "APD_OVER_PNACOR_MEAN",  ix, iy,  apdpn_apdpn[iAPDoPNACor][iMean] );
	  setVal( "APD_OVER_PNACOR_RMS",   ix, iy,  apdpn_apdpn[iAPDoPNACor][iRMS] );
	  setVal( "APD_OVER_PNACOR_M3",    ix, iy,  apdpn_apdpn[iAPDoPNACor][iM3] );  
	  setVal( "APD_OVER_PNACOR_NEVT",  ix, iy,  apdpn_apdpn[iAPDoPNACor][iNevt] );
	  setVal( "APD_OVER_PNBCOR_MEAN",  ix, iy,  apdpn_apdpn[iAPDoPNBCor][iMean] );
	  setVal( "APD_OVER_PNBCOR_RMS",   ix, iy,  apdpn_apdpn[iAPDoPNBCor][iRMS] );
	  setVal( "APD_OVER_PNBCOR_M3",    ix, iy,  apdpn_apdpn[iAPDoPNBCor][iM3] );  
	  setVal( "APD_OVER_PNBCOR_NEVT",  ix, iy,  apdpn_apdpn[iAPDoPNBCor][iNevt] );
	  setVal( "APD_OVER_PNCOR_MEAN",   ix, iy,  apdpn_apdpn[iAPDoPNCor][iMean] );
	  setVal( "APD_OVER_PNCOR_RMS",    ix, iy,  apdpn_apdpn[iAPDoPNCor][iRMS] );
	  setVal( "APD_OVER_PNCOR_M3",     ix, iy,  apdpn_apdpn[iAPDoPNCor][iM3] );  
	  setVal( "APD_OVER_PNCOR_NEVT",   ix, iy,  apdpn_apdpn[iAPDoPNCor][iNevt] );

	  setVal( "APD_OVER_APD_MEAN", ix, iy,  apdpn_apdpn[iAPDoAPD][iMean] );
	  setVal( "APD_OVER_APD_RMS",  ix, iy,  apdpn_apdpn[iAPDoAPD][iRMS] );
	  setVal( "APD_OVER_APD_M3",   ix, iy,  apdpn_apdpn[iAPDoAPD][iM3] );  
	  setVal( "APD_OVER_APD_NEVT",  ix, iy,  apdpn_apdpn[iAPDoAPD][iNevt] );
	  setVal( "APD_OVER_APDA_MEAN", ix, iy,  apdpn_apdpn[iAPDoAPDA][iMean] );
	  setVal( "APD_OVER_APDA_RMS",  ix, iy,  apdpn_apdpn[iAPDoAPDA][iRMS] );
	  setVal( "APD_OVER_APDA_M3",   ix, iy,  apdpn_apdpn[iAPDoAPDA][iM3] ); 
	  setVal( "APD_OVER_APDA_NEVT",  ix, iy,  apdpn_apdpn[iAPDoAPDA][iNevt] );
	  setVal( "APD_OVER_APDB_MEAN", ix, iy,  apdpn_apdpn[iAPDoAPDB][iMean] );
	  setVal( "APD_OVER_APDB_RMS",   ix, iy,  apdpn_apdpn[iAPDoAPDB][iRMS] );
	  setVal( "APD_OVER_APDB_M3",    ix, iy,  apdpn_apdpn[iAPDoAPDB][iM3] );  
	  setVal( "APD_OVER_APDB_NEVT",  ix, iy,  apdpn_apdpn[iAPDoAPDB][iNevt] );
	  // JM
	  setVal( "SHAPE_COR_APD",          ix, iy,  apdpn_shapeCorAPD );
	
	  // NEW GHM 08/06
	  setVal( "TIME_MEAN",          ix, iy,  apdpn_apdpn[iTime][iMean] );
	  setVal( "TIME_RMS",           ix, iy,  apdpn_apdpn[iTime][iRMS]  );
	  setVal( "TIME_M3",            ix, iy,  apdpn_apdpn[iTime][iM3]   );
	  setVal( "TIME_NEVT",          ix, iy,  apdpn_apdpn[iTime][iNevt] );

	}
    
      //      cout<< "LED APD FILLED "<< endl ; //TESTJULIE
     
      //
      // PN primitives
      //
      t_name = lmfLaserName( ME::iLmfLEDPnPrim, _type, _color );

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
		  
	      //if( jj==1 ) assert( pn_moduleID==module_ ); what is this for?
	      module_ = pn_moduleID;
	      assert( pn_pnID==jj );
	      
	      // get the PN number
	      pair<int,int> memPn_ = ME::pn( _lmr, module_, (ME::PN)jj );
	      if( _isBarrel )
		{
		  id1_ = _sm;
		  id2_ = memPn_.second;
		}
	      else
		{
		  int dee_ = MEEEGeom::dee( _lmr );

		  id1_ = dee_;
		  id2_ = (jj+1)*100+memPn_.second;
		}
	      
	      //	      cout <<"  pn prim " <<id1_<<"  " <<id2_<<"  " <<_lmr<<"  " <<module_<<"  " << jj<< endl; // TESTJULIE

	      if( _isBarrel )
		{
		  channelView_ = iEB_LM_PN;
		}
	      else
		{
		  channelView_ = iEE_LM_PN;
		}
	      logic_id_ = logicId( channelView_, id1_, id2_ );
	      int pnflag=0;
	      if(pn_PN[iMean]>200. &&pn_PN[iMean]<5000.) pnflag=1;
	      i_t[t_name+separator+"LOGIC_ID"] = logic_id_;
	      f_t[t_name+separator+"MEAN"]  = pn_PN[iMean];
	      i_t[t_name+separator+"FLAG"]  = pnflag;
	      f_t[t_name+separator+"RMS"]   = pn_PN[iRMS];
	      f_t[t_name+separator+"M3"]  = pn_PN[iM3];   
	      f_t[t_name+separator+"NEVT"]  = pn_PN[iNevt];    
	      f_t[t_name+separator+"PNA_OVER_PNB_MEAN"]  = (jj==0) ? pn_PNoPNB[iMean] : pn_PNoPNA[iMean];
	      f_t[t_name+separator+"PNA_OVER_PNB_RMS" ]  = (jj==0) ? pn_PNoPNB[iRMS]  : pn_PNoPNA[iRMS];
	      f_t[t_name+separator+"PNA_OVER_PNB_M3"]  = (jj==0) ? pn_PNoPNB[iM3] : pn_PNoPNA[iM3];
	      f_t[t_name+separator+"SHAPE_COR_PN"]  = pn_shapeCorPN;   

	      
	      t_t[t_name]->Fill();
	      
	    }
	  //      cout << "Module=" << module_ << "\tPNA=" << pn_[0] << "\tPNB=" << pn_[1] << endl;
	  
	  // 	  if( _isBarrel )
	  // 	    jentry += 4;
	  // 	  else
	  // 	    jentry += 2;
	  jentry += 2;
	}
      
      logic_id_  = logicId( iECAL_LMR, _lmr );

      

      //      cout<< "LED PN FILLED "<< endl ; //TESTJULIE
     
      //
      // Laser Run
      //
      t_name = lmfLaserName( ME::iLmfLEDRun, _type );
      //cout << "Fill "<< t_name << endl;
      i_t[t_name+separator+"LOGIC_ID"]       = logic_id_; 
      i_t[t_name+separator+"NEVENTS"]        =  _events;
      i_t[t_name+separator+"QUALITY_FLAG"]   = 1;                // fixme
      t_t[t_name]->Fill();
  

      //      cout<< "LED RUN FILLED "<< endl ; //TESTJULIE
    }

  //
  // Laser Run IOV
  //
  t_name = "LMF_RUN_IOV";
  //cout << "Fill "<< t_name << endl;
  i_t[t_name+separator+"TAG_ID"]        = 0;       // fixme
  i_t[t_name+separator+"SUB_RUN_NUM"]   = _run;      // fixme
  i_t[t_name+separator+"SUB_RUN_START_LOW" ] = ME::time_low( _ts_beg );
  i_t[t_name+separator+"SUB_RUN_START_HIGH"] = ME::time_high( _ts_beg );
  i_t[t_name+separator+"SUB_RUN_END_LOW"   ] = ME::time_low( _ts_end );
  i_t[t_name+separator+"SUB_RUN_END_HIGH"  ] = ME::time_high( _ts_end );
  i_t[t_name+separator+"DB_TIMESTAMP_LOW"  ] = ME::time_low( _ts ); 
  i_t[t_name+separator+"DB_TIMESTAMP_HIGH" ] = ME::time_high( _ts );
  c_t[t_name+separator+"SUB_RUN_TYPE"]  = "LASER BEAM"; //fixme
  t_t[t_name]->Fill();

}

void
MELaserPrim::writeHistograms()
{
  if( !init_ok ) return;

  out_file = new TFile( _outfile, "RECREATE" );
  //  out_file->cd();

  map< TString, TH2* >::iterator it;

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

  map< TString, TTree* >::iterator it_t;
  for( it_t=t_t.begin(); it_t!=t_t.end(); it_t++ )
    {
      it_t->second->Write();
      delete it_t->second;
    }

  cout << "Closing " << _outfile << endl;
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
  
  if( _type==ME::iLaser ){
    if( apdpn_file!=0 )
      {
	cout << "Closing apdpn_file " << endl;
	apdpn_file->Close();
	delete apdpn_file;
	apdpn_file = 0;
      }
    if( apdpnabfit_file!=0)
      {
	cout << "Closing apdpn_file " << endl;
	apdpnabfit_file->Close();
	delete apdpnabfit_file;
	apdpnabfit_file = 0;
      }
    if( apdpnabfix_file!=0 )
      {
	cout << "Closing apdpn_file " << endl;
	apdpnabfix_file->Close();
	delete apdpnabfix_file;
	apdpnabfix_file = 0;
      }
    if( ab_file!=0 )
      {
	cout << "Closing ab_file " << endl;
	ab_file->Close();
	delete ab_file;
	ab_file = 0;
      }
    if( mtq_file!=0 )
      {
	cout << "Closing mtq_file " << endl;
	mtq_file->Close();
	delete mtq_file;
	mtq_file = 0;
      }
  } else if( _type==ME::iLED ){
    if( apdpn_file!=0 )
      {
	cout << "Closing apdpn_file " << endl;
	apdpn_file->Close();
	delete apdpn_file;
	apdpn_file = 0;
      }
  }
  else{
    if( tpapd_file!=0 )
      {
	cout << "Closing tpapd_file " << endl;
	tpapd_file->Close();
	delete tpapd_file;
	tpapd_file = 0;
      }
  }
}

void 
MELaserPrim::print( ostream& o )
{
  o << "DCC/SM/side/type/color/run/ts " << _dcc << "/" << _sm << "/" << _side << "/" 
    << _type << "/" << _color << "/" << _run << "/" << _ts << endl;
  
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
  else if( type==ME::iLED )
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
      str = "LMF_LED";
      switch( table )
	{
	case ME::iLmfLEDRun:      str  = "LMF_RUN";                 break; 
	case ME::iLmfLEDPrim:     str += colstr; str += "_PRIM";    break; 
	case ME::iLmfLEDPnPrim:   str += colstr; str += "_PN_PRIM"; break; 
	default:  abort();
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
  else if ( _type==ME::iLED ) name_=_LEDprimStr+name;
 
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
  else if ( _type==ME::iLED ) name_=_LEDprimStr+name;
 
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
      cout << key_ << endl;
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
    
