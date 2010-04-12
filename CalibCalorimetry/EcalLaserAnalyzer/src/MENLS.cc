#define MENLS_cxx
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MENLS.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEGeom.h"
#include <cassert>
#include <cstdlib>


TString MENLS::nls_histName[MENLS::iSize_nls] = { "MEAN", "RMS", "NEVT", "FLAG", "NORM", "ENORM"};
TString MENLS::nlsref_histName[MENLS::iSizeRef_nls] = { "RUN_NUM","LB_NUM","START_LOW","START_HIGH"};
//TString MENLS::nls_varUnit[MENLS::iSizeArray_nls][MENLS::iSize_nls] =  
//  {    {"", "", "", "", "", ""}
//  };

TString MENLS::separator = "__";

//ClassImp( MENLS )

MENLS::MENLS(  ME::Header header, ME::Settings settings, 
			    const char* outfile )
: init_ok(false), _isBarrel(true), _outfile(outfile)
{
  ixmin      =0;
  ixmax      =0;
  iymin      =0;
  iymax      =0;
  _debug=false;

  // TODO:
  // Récuperer cette info des fichiers de primitives facon gautier
  //================================================================
  if(_debug) cout<<"Entering MENLS constructor"<< endl;

  _dcc    = header.dcc;
  _side   = header.side;
  _run    = header.run;
  _lb     = header.lb;
  _events = header.events;
  _ts     = header.ts_beg;
  _ts_beg = header.ts_beg;
  _ts_end = header.ts_end;

  _type     = ME::iLaser;
  _color    = settings.wavelength;
  _power    = settings.power;
  _filter   = settings.filter;
  _delay    = settings.delay;
  _mgpagain = settings.mgpagain; 
  _memgain  = settings.memgain; 

  if(_debug) cout<<"MENLS -- dcc="<<_dcc<<" side="<< _side<<" color="<< _color<< endl;

  _nlsStr     = lmfNLSName( ME::iLmfNLS, _color )+separator;
  _refStr     = lmfNLSName( ME::iLmfNLSRef, _color )+separator;

  if(_debug) cout<<"MENLS -- nlsStr="<<_nlsStr<< endl;
  if(_debug) cout<<"MENLS -- refStr="<<_refStr<< endl;
  
  _lmr = ME::lmr( _dcc, _side );

  ME::regionAndSector( _lmr, _reg, _sm, _dcc, _side );    
  _isBarrel = (_reg==ME::iEBM || _reg==ME::iEBP);
  _sectorStr  = ME::smName( _lmr );
  _regionStr  = _sectorStr;
  _regionStr += "_"; _regionStr  += _side;
  
  init();
  if(_debug) cout<<"MENLS -- Booking histograms"<< endl;
  bookHistograms();
  if(_debug) cout<<"MENLS -- ... done "<< endl;
  if(_debug) cout<<"MENLS -- Filling run info"<< endl;
  fillRun();
  if(_debug) cout<<"MENLS -- ... done "<< endl;

  //fillHistograms();
  //writeHistograms();
}

TString
MENLS::channelViewName( int iname )
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
MENLS::logicId( int channelView, int id1, int id2 )
{
  assert( channelView>=iECAL && channelView<iSize_cv );
  return 1000000*channelView + 10000*id1 + id2;
}

bool 
MENLS::getViewIds( int logic_id, int& channelView, int& id1, int& id2 )
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
MENLS::init()
{
}

void
MENLS::bookHistograms()
{
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

    }
  else   
    {
      ixmin=1;
      ixmax=101;
      nx   =ixmax-ixmin;
      iymin=1;
      iymax=101;
      ny   =iymax-iymin;

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
  t_name = "LMF_CLS_ID";
  addBranchI( t_name, "TAG_ID"         );
  addBranchI( t_name, "RUN_NUM"        );
  addBranchI( t_name, "LB_NUM"         );
  addBranchI( t_name, "START_LOW"      );
  addBranchI( t_name, "START_HIGH"     );
  addBranchI( t_name, "END_LOW"        );
  addBranchI( t_name, "END_HIGH"       );
  addBranchI( t_name, "DB_TIMESTAMP_LOW"    );
  addBranchI( t_name, "DB_TIMESTAMP_HIGH"   );
  addBranchC( t_name, "RUN_TYPE"   );


  bookHistoI( _nlsStr, "LOGIC_ID" );
  bookHistoI( _nlsStr, "FLAG" );
  bookHistoF( _nlsStr, "MEAN" );
  bookHistoF( _nlsStr, "RMS" );
  bookHistoF( _nlsStr, "NEVT" );
  bookHistoF( _nlsStr, "NORM" );
  bookHistoF( _nlsStr, "ENORM" );

  bookHistoI( _refStr, "RUN_NUM" );
  bookHistoI( _refStr, "LB_NUM" );
  bookHistoI( _refStr, "START_LOW"   );
  bookHistoI( _refStr, "START_HIGH"  );


  // FIXME: add this per channel...
  // Ref Laser Run IOV
  //
  // t_name = "LMF_RUN_IOV_REF";
  //   addBranchI( t_name, "TAG_ID"         );
  //   addBranchI( t_name, "SUB_RUN_NUM"    );
  //   addBranchI( t_name, "SUB_RUN_START_LOW"   );
  //   addBranchI( t_name, "SUB_RUN_START_HIGH"  );
  //   addBranchI( t_name, "SUB_RUN_END_LOW"     );
  //   addBranchI( t_name, "SUB_RUN_END_HIGH"    );
  //   addBranchI( t_name, "DB_TIMESTAMP_LOW"    );
  //   addBranchI( t_name, "DB_TIMESTAMP_HIGH"   );
  //   addBranchC( t_name, "SUB_RUN_TYPE"   );
  
  
  init_ok=true;
  
}

void
MENLS::fillChannel(int ieta, int iphi, int ivar, float val )
{

 if(_debug)  cout<< "MENLS -- Entering fillChannel "<<ivar<<" "<<val<< endl;
  
  if( !init_ok ) return;
  assert( ivar < iSize_nls );
  
  int channelView_(0);
  int id1_(0), id2_(0);
  int logic_id_(0);
   
	  
  int ix(0);
  int iy(0);
  if( _isBarrel )
    {
      // Barrel, global coordinates
      id1_ = _sm;   
      MEEBGeom::XYCoord xy_ = MEEBGeom::localCoord( ieta, iphi );
      ix = xy_.first;
      iy = xy_.second;
      id2_ = MEEBGeom::crystal_channel( ix, iy ); 
      channelView_ = iEB_crystal_number;
    }
  else
    {
      // EndCaps, global coordinates
      id1_ = iphi;
      id2_ = ieta; 
      ix = id1_;
      iy = id2_;
      channelView_ = iEE_crystal_number;
    }
  
  logic_id_ = logicId( channelView_, id1_, id2_ );

  if( ivar == iFlag ){

    int flag=int(val);

    if(_debug) cout << "MENLS -- Filling flag "<<flag<<" var:"<<ivar<<" "<< nls_histName[ivar]<< endl;
    setInt(  nls_histName[ivar],  ix, iy,  flag );
    if(_debug) cout << "...done"<< endl;
    if(_debug) cout << "MENLS -- Filling logic_id "<<logic_id_<< endl;
    setInt( "LOGIC_ID",           ix, iy,  logic_id_ );
    if(_debug) cout << "... done"<< endl;
  }else{
    if(_debug) cout << "MENLS -- Filling val "<<val<<" var:"<<ivar<<" "<< nls_histName[ivar]<<endl;
    setVal( nls_histName[ivar], ix, iy, val );
  }
    
}

void
MENLS::fillRefChannel( int ieta, int iphi, int runnum, int lbnum, int startlow, int starthigh ){
  
  if( !init_ok ) return;
  
  int channelView_(0);
  int id1_(0), id2_(0);
   
	  
  int ix(0);
  int iy(0);
  if( _isBarrel )
    {
      // Barrel, global coordinates
      id1_ = _sm;   
      MEEBGeom::XYCoord xy_ = MEEBGeom::localCoord( ieta, iphi );
      ix = xy_.first;
      iy = xy_.second;
      id2_ = MEEBGeom::crystal_channel( ix, iy ); 
      channelView_ = iEB_crystal_number;
    }
  else
    {
      // EndCaps, global coordinates
      id1_ = iphi;
      id2_ = ieta; 
      ix = id1_;
      iy = id2_;
      channelView_ = iEE_crystal_number;
    }


    setIntRef( nlsref_histName[iRunRef],               ix, iy,    runnum );
    setIntRef( nlsref_histName[iLBRef],                ix, iy,     lbnum );
    setIntRef( nlsref_histName[iTimeLowRef],           ix, iy,  startlow );
    setIntRef( nlsref_histName[iTimeHighRef],          ix, iy, starthigh );

    
}

void
MENLS::fillRun()
{

 if(_debug)  cout<< "MENLS -- Entering fillRun "<<endl;

  if( !init_ok ) return;

  //
  //Run
  //

  int logic_id_(0);
  logic_id_  = logicId( iECAL_LMR, _lmr );
  if(_debug) cout<< "MENLS -- logic_id "<< logic_id_  << " events "<<_events<<endl;
  TString t_name; 
  t_name = "LMF_RUN_DAT";
  if(_debug) cout << "Filling "<< t_name <<endl;
  i_t[t_name+separator+"LOGIC_ID"]       = logic_id_; 
  i_t[t_name+separator+"NEVENTS"]        =  _events;
  i_t[t_name+separator+"QUALITY_FLAG"]   = 1;                // fixme
  t_t[t_name]->Fill();
  
  if(_debug) cout << " ... done" << endl;
  //
  // Laser Run IOV
  //

  t_name = "LMF_CLS_ID";
  
  if(_debug) cout << "Filling "<< t_name <<endl;
  //  i_t[t_name+separator+"TAG_ID"]        = 0;       // fixme
  i_t[t_name+separator+"RUN_NUM"]   = _run;      // fixme
  i_t[t_name+separator+"LB_NUM"]   = _lb;      // fixme
  i_t[t_name+separator+"START_LOW" ] = ME::time_low( _ts_beg );
  i_t[t_name+separator+"START_HIGH"] = ME::time_high( _ts_beg );
  i_t[t_name+separator+"END_LOW"   ] = ME::time_low( _ts_end );
  i_t[t_name+separator+"END_HIGH"  ] = ME::time_high( _ts_end );
  i_t[t_name+separator+"DB_TIMESTAMP_LOW"  ] = ME::time_low( _ts ); 
  i_t[t_name+separator+"DB_TIMESTAMP_HIGH" ] = ME::time_high( _ts );
  c_t[t_name+separator+"RUN_TYPE"]  = "LASER CRAFT09"; //fixme
  t_t[t_name]->Fill();
  if(_debug) cout << " ... done" << endl;

}

void
MENLS::writeHistograms()
{

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

  //  cout << "Closing " << _outfile << endl;
  out_file->Close();
  delete out_file;
  out_file=0;
}

MENLS::~MENLS()
{
}

void 
MENLS::print( ostream& o )
{
  o << "DCC/SM/side/type/color/run/ts " << _dcc << "/" << _sm << "/" << _side << "/" 
    << _type << "/" << _color << "/" << _run << "/" << _ts << endl;
  
//   for( int ix=ixmin; ix<ixmax; ix++ )
//     {
//       for( int iy=iymin; iy<iymax; iy++ )
// 	{
// 	  int flag     = getInt( _nlsstr+"FLAG", ix, iy );
// 	  if( flag==0 ) continue;
// 	  int logic_id = getInt( _nlsstr+"LOGIC_ID", ix, iy );
// 	  float   apd = getVal( _nlsstr+"MEAN", ix, iy );
// 	  o << "Crystal ieta=" << ix << "\tiphi=" << iy << "\tlogic_id=" << logic_id << "\tAPD=" << apd <<  endl;
// 	}
//     }
}

TString 
MENLS::lmfNLSName( int table, int color )
{

  TString str("LMF_ERROR");
  if( table<0 || table>=ME::iSizeLmf )  return str;
  if( color<0 || color>=ME::iSizeC )    return str;

  
  TString colstr;
  switch( color )
    {
    case ME::iBlue:   colstr = "_BLUE"; break;
    case ME::iGreen:  colstr = "_GREEN"; break;
    case ME::iRed:    colstr = "_RED";  break;
    case ME::iIRed:   colstr = "_IRED";  break;
    default:  abort();
    }
  str = "LMF_CLS";
  switch( table )
    {
    case ME::iLmfNLSRun:      str  = "_RUN";                 break; 
    case ME::iLmfNLS:     str += colstr; break; 
    case ME::iLmfNLSRef:  str +="_ID_REF";  str += colstr; break; 
    default:  abort();
    }

  str += "_DAT";
  return str;
}

void
MENLS::addBranchI( const char* t_name_, const char* v_name_ )
{
  TString slashI("/i"); // Warning: always unsigned
  TString t_name(t_name_);
  TString v_name(v_name_);
  if( t_t.count(t_name)==0 ) t_t[t_name] = new TTree(t_name, t_name);
  t_t[t_name]->Branch(v_name, &i_t[t_name+separator+v_name],v_name+slashI);  
}

void
MENLS::addBranchF( const char* t_name_, const char* v_name_ )
{
  TString slashF("/F");
  TString t_name(t_name_);
  TString v_name(v_name_);
  if( t_t.count(t_name)==0 ) t_t[t_name] = new TTree(t_name, t_name);
  t_t[t_name]->Branch(v_name, &f_t[t_name+separator+v_name],v_name+slashF);
}

void
MENLS::addBranchC( const char* t_name_, const char* v_name_ )
{
  TString slashC("/C");
  TString t_name(t_name_);
  TString v_name(v_name_);
  if( t_t.count(t_name)==0 ) t_t[t_name] = new TTree(t_name, t_name);
  t_t[t_name]->Branch(v_name, &c_t[t_name+separator+v_name],v_name+slashC);
}

void 
MENLS::bookHistoI( const char* h_name_, const char* v_name_ )
{
  TString i_name = TString(h_name_)+TString(v_name_);
  TH2* h_ = new TH2I(i_name,i_name,nx,ixmin,ixmax,ny,iymin,iymax);
  setHistoStyle( h_ );
  i_h[i_name] = h_;
}

void 
MENLS::bookHistoF( const char* h_name_, const char* v_name_ )
{
  TString d_name = TString(h_name_)+TString(v_name_);
  TH2* h_ = new TH2F(d_name,d_name,nx,ixmin,ixmax,ny,iymin,iymax);
  setHistoStyle( h_ );
  f_h[d_name] = h_;
}


bool    
MENLS::setInt( const char* name, int ix, int iy, int ival )
{
  TString name_;
  name_=_nlsStr+name;
   
  //  cout<<"setInt "<< name_<<" "<<ix<<" "<<iy<<" "<<ival<< endl;
  int _ival = getInt( name_, ix, iy );
   
  //  cout<<"setInt2 "<< _ival<< endl;
  assert( _ival!=-99 );
  if( _ival!=0 ) return false; 

  TH2I* h_ = (TH2I*) i_h[name_];
  assert( h_!=0 );
  h_->Fill( ix+0.5, iy+0.5, ival );

  return true;
}
bool    
MENLS::setIntRef( const char* name, int ix, int iy, int ival )
{
  TString name_;
  name_=_refStr+name;
   
  //  cout<<"setInt "<< name_<<" "<<ix<<" "<<iy<<" "<<ival<< endl;
  int _ival = getInt( name_, ix, iy );
   
  //  cout<<"setInt2 "<< _ival<< endl;
  assert( _ival!=-99 );
  if( _ival!=0 ) return false; 

  TH2I* h_ = (TH2I*) i_h[name_];
  assert( h_!=0 );
  h_->Fill( ix+0.5, iy+0.5, ival );

  return true;
}

bool    
MENLS::setVal( const char* name, int ix, int iy, float val )
{
  TString name_;
  name_=_nlsStr+name;
   
  float _val = getVal( name_, ix, iy );
  assert( _val!=-99 );
  if( _val!=0 ) return false; 

  TH2F* h_ = (TH2F*) f_h[name_];
  assert( h_!=0 );
  
  h_->Fill( ix+0.5, iy+0.5, val );

  return true;
}
bool    
MENLS::setValRef( const char* name, int ix, int iy, float val )
{
  TString name_;
  name_=_refStr+name;
   
  float _val = getVal( name_, ix, iy );
  assert( _val!=-99 );
  if( _val!=0 ) return false; 

  TH2F* h_ = (TH2F*) f_h[name_];
  assert( h_!=0 );
  
  h_->Fill( ix+0.5, iy+0.5, val );

  return true;
}


Int_t    
MENLS::getInt( const char* name, int ix, int iy )
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
MENLS::getVal( const char* name, int ix, int iy )
{
  if(_debug) cout<< "MENLS -- Inside getVal "<< name<<" "<<ix<<" "<<iy<< endl; 
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
MENLS::setInt( const char* tname, const char* vname, int ival )
{
  TString key_(tname); key_ += separator; key_ += vname;
  assert( i_t.count(key_)==1 );
  i_t[key_] = ival;
  return true;
}

bool
MENLS::setVal( const char* tname, const char* vname, float val )
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
MENLS::fill( const char* tname )
{
  TString key_( tname );
  assert( t_t.count(key_)==1 );
  t_t[key_] -> Fill();
  return true;
}

void 
MENLS::setHistoStyle( TH1* h )
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
MENLS::refresh()
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

TString MENLS::nlsVarName(int var){
  assert (var<iSize_nls);
  return nls_histName[var];

}
TString MENLS::nlsRefVarName(int var){
  assert (var<iSizeRef_nls);
  return nlsref_histName[var];
}
