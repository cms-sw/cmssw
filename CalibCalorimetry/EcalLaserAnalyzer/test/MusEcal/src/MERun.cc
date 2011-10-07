#define MERun_cxx
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cassert>
using namespace std;

#include "MERun.hh"
#include "MENLS.hh"
#include "../../interface/MELaserPrim.h"
#include "../../interface/MEGeom.h"

ClassImp( MERun )

MERun::MERun( ME::Header header, ME::Settings settings, TString fname )
: _header( header ), _settings( settings ), _fname(fname), _file(0), 
  _nlsfile(0), pn_t(0), mtq_t(0)
{
  _type = _settings.type;
  _color = _settings.wavelength;

  if( _type ==ME::iTestPulse ) _color=0;

  if( settings.type== ME::iLaser ) _fnlsname = ME::rootNLSFileName( header, settings );
 
  // cout<<" Inside MERun "<< endl;
}

MERun::~MERun()
{
  closeLaserPrimFile();
}

void
MERun::closeLaserPrimFile()
{

  if( _file ) _file->Close();
  delete _file;
  _file = 0;
  _h.clear();
  
}

TFile*
MERun::laserPrimFile( bool refresh )
{
  if( refresh )
    closeLaserPrimFile();

  if( _file==0 )
    {
      FILE *test;
      test= fopen( _fname, "r" );
      if(test)
	{
	  _file = TFile::Open( _fname );
	  fclose( test );
	}
      assert( _file!=0 );
    }
  return _file;
}

TH2*
MERun::APDHist( int var )
{
  //  assert( var>=0 && var<ME::iSizeAPD );
  // get an histogram in the LmfLaserPrim table
  int table(0);
  TString varName;
  if( _type==ME::iLaser )
    {
      table = ME::iLmfLaserPrim;
      varName = ME::APDPrimVar[var];
    }
  else if( _type==ME::iTestPulse  )
    {
      table = ME::iLmfTestPulsePrim;
      varName = ME::TPAPDPrimVar[var];
    }
  else if( _type==ME::iLED )
    {
      table = ME::iLmfLEDPrim;
      varName = ME::APDPrimVar[var];
    }

  TH2* h(0);

  TFile* f = laserPrimFile();
  
  if( f==0 ) return h;
  
  TString histName = MELaserPrim::lmfLaserName( table, _type, _color );
  histName += MELaserPrim::separator;
  histName += varName;
  if( _h.count(varName)==0 )
    {
      h = (TH2*) f->Get( histName );
      if( h!=0 ) _h[varName] = h;
      if( varName == "LOGIC_ID" )
	// if( varName == "MEAN" )
	{
	  TAxis* ax = h->GetXaxis();
	  TAxis* ay = h->GetYaxis();
      
	  cout << "X axis Nbins=" << ax->GetNbins() 
	       << " first=" << ax->GetFirst()
	       << " lowedge(first)=" << ax->GetBinLowEdge(ax->GetFirst())
	       << " last="  << ax->GetLast()
	       << " lowedge(last)=" << ax->GetBinLowEdge(ax->GetLast())
	       << endl;
	  cout << "Y axis Nbins=" << ay->GetNbins() 
	       << " first=" << ay->GetFirst()
	       << " lowedge(first)=" << ay->GetBinLowEdge(ay->GetFirst())
	       << " last="  << ay->GetLast()
	       << " lowedge(last)=" << ay->GetBinLowEdge(ay->GetLast())
	       << endl;
	}
    }
  else
    {
      h = _h[varName];
    }
  return h;
}

TTree*
MERun::PNTable()
{
  TFile* f = laserPrimFile();
  if( f==0 ) return 0;

  unsigned int size_(0);
  int table(0);
  if( _type==ME::iLaser )
    {
      table = ME::iLmfLaserPnPrim;
      size_ = ME::iSizePN;
    }
  else if( _type==ME::iTestPulse )
    {
      table = ME::iLmfTestPulsePnPrim;
      size_ = ME::iSizeTPPN;
    }
  else if( _type==ME::iLaser )
    {
      table = ME::iLmfLEDPnPrim;
      size_ = ME::iSizePN;
    }

  TString tableName = MELaserPrim::lmfLaserName( table, _type, _color );
  // don't cache the pn_t pointer, it's dangerous
  pn_i.clear();
  pn_d.clear();
 
  pn_t = (TTree*) f->Get(tableName);
  if( pn_t==0){
    cout<<" PNTABLENAME: " <<tableName<<" "<<table<<" "<<_type<<" "<< _settings.wavelength<< endl;
    f->ls();
    cout<<" filename: " << f->GetName()<< endl;
  }
  assert( pn_t!=0 );
  TString vname;
  vname = "LOGIC_ID"; pn_t->SetBranchAddress( vname, &pn_i[vname] );
  vname = "FLAG";     pn_t->SetBranchAddress( vname, &pn_i[vname] );
  for( unsigned int ii=1; ii<size_; ii++ )
    {
      vname = ME::PNPrimVar[ii]; pn_t->SetBranchAddress( vname, &pn_d[vname] );
    }
  return pn_t;
}

TTree*
MERun::MTQTable()
{

  TFile* f = laserPrimFile();
  if( f==0 ) return 0;
  int table = ME::iLmfLaserPulse;
  TString tableName = MELaserPrim::lmfLaserName( table, _type, _color );
  // don't cache the mtq_t pointer, it's dangerous
  mtq_i.clear();
  mtq_d.clear();
  mtq_t = (TTree*) f->Get(tableName);
  assert( mtq_t!=0 );
  TString vname;
  vname = "LOGIC_ID";   mtq_t->SetBranchAddress( vname, &mtq_i[vname] );
  vname = "FIT_METHOD"; mtq_t->SetBranchAddress( vname, &mtq_i[vname] );
  for( int ii=ME::iMTQ_FIT_METHOD+1; ii<ME::iSizeMTQ; ii++ )
    {
      vname = ME::MTQPrimVar[ii]; mtq_t->SetBranchAddress( vname, &mtq_d[vname] );
    }
  return mtq_t;
}
TFile*
MERun::NLSFile( bool& doesExist , bool refresh )
{
  doesExist=true;

  // cout << "MERun -- Entering NLSFile ... "<<doesExist<<" "<< refresh<< " "<<_fnlsname<<endl;

  if( refresh )
    closeNLSFile();
  
  if( _nlsfile==0 )
    {
      FILE *test;
      test= fopen( _fnlsname, "r" );
      if(test)
	{
	  _nlsfile = TFile::Open( _fnlsname );
	  fclose( test );
	}else{
	  doesExist=false;
	}
    }
  //cout << "MERun -- Quitting NLSFile ... "<< doesExist<<"  " << _fnlsname<< endl;
  return _nlsfile;
}
void
MERun::closeNLSFile()
{
  //  cout << "MERun -- Closing NLSFile ... "<< endl;

  if( _nlsfile ) _nlsfile->Close();
  delete _nlsfile;
  _nlsfile = 0;
  _hnls.clear();
  _hnlsref.clear();
  //  cout <<"MERun -- ... done"<< endl;
  
}


TH2*
MERun::NLSHist( int var , bool& doesExist )
{
  doesExist=true;

  //  cout << "MERun -- Entering NLSHist "<<var<<" "<< doesExist<< endl;

  assert( var>=0 && var<MENLS::iSize_nls);

  // get an histogram in the lmfNLSPrim table
  int table(0);
  TString varName;
  assert ( _type==ME::iLaser );
    
  table = ME::iLmfNLS;
  varName = MENLS::nlsVarName(var);
  
  TH2* h(0);

  TFile* f = NLSFile( doesExist, false );
  
  //  cout << "MERun -- NLSHist doesExist="<< doesExist<< endl;

  if(!doesExist) return h;
  
  if( f==0 ){
    doesExist=false;
    return h;
  }
  
  TString histName = MENLS::lmfNLSName( table, _settings.wavelength );
  histName += MENLS::separator;
  histName += varName;
  //  cout << "MERun -- doesExist:"<< doesExist<< " histName: "<<histName<<endl;

  if( _hnls.count(varName)==0 )
    {
      h = (TH2*) f->Get( histName );
      if( h!=0 ) {
	_hnls[varName] = h;
	//	cout << " h exists"<<endl;
      }
      else cout << "histo = 0 "<< endl;
      

      if( varName == "LOGIC_ID" )
	// if( varName == "MEAN" )
	{
	  TAxis* ax = h->GetXaxis();
	  TAxis* ay = h->GetYaxis();
      
	  cout << "X axis Nbins=" << ax->GetNbins() 
	       << " first=" << ax->GetFirst()
	       << " lowedge(first)=" << ax->GetBinLowEdge(ax->GetFirst())
	       << " last="  << ax->GetLast()
	       << " lowedge(last)=" << ax->GetBinLowEdge(ax->GetLast())
	       << endl;
	  cout << "Y axis Nbins=" << ay->GetNbins() 
	       << " first=" << ay->GetFirst()
	       << " lowedge(first)=" << ay->GetBinLowEdge(ay->GetFirst())
	       << " last="  << ay->GetLast()
	       << " lowedge(last)=" << ay->GetBinLowEdge(ay->GetLast())
	       << endl;
	}
    }
  else
    {
      h = _hnls[varName];
    }
  
  //  cout << " MERun -- quitting NLSHist "<<h->GetCellContent(1,1)<< endl;
  return h;


}

TH2*
MERun::NLSRefHist( int var , bool& doesExist )
{
  doesExist=true;
  //  cout << "MERun -- Entering NLSRefHist "<<var<<" "<< doesExist<< endl;

  assert( var>=0 && var<MENLS::iSizeRef_nls );

  // get an histogram in the lmfNLSPrim table
  int table(0);
  TString varName;
  assert ( _type==ME::iLaser );
    
  table = ME::iLmfNLSRef;
  varName = MENLS::nlsRefVarName(var);
  
  TH2* h(0);

  TFile* f = NLSFile( doesExist , false );
  
  if(!doesExist) return h;
  
  if( f==0 ){
    doesExist=false;
    return h;
  }
  
  TString histName = MENLS::lmfNLSName( table, _settings.wavelength );
  histName += MENLS::separator;
  histName += varName;
  //  cout << "MERun -- doesExist:"<< doesExist<<" histName:"<<histName<< endl;

  if( _hnlsref.count(varName)==0 )
    {
      h = (TH2*) f->Get( histName );
      if( h!=0 ) _hnlsref[varName] = h;
      if( varName == "LOGIC_ID" )
	// if( varName == "MEAN" )
	{
	  TAxis* ax = h->GetXaxis();
	  TAxis* ay = h->GetYaxis();
      
	  cout << "X axis Nbins=" << ax->GetNbins() 
	       << " first=" << ax->GetFirst()
	       << " lowedge(first)=" << ax->GetBinLowEdge(ax->GetFirst())
	       << " last="  << ax->GetLast()
	       << " lowedge(last)=" << ax->GetBinLowEdge(ax->GetLast())
	       << endl;
	  cout << "Y axis Nbins=" << ay->GetNbins() 
	       << " first=" << ay->GetFirst()
	       << " lowedge(first)=" << ay->GetBinLowEdge(ay->GetFirst())
	       << " last="  << ay->GetLast()
	       << " lowedge(last)=" << ay->GetBinLowEdge(ay->GetLast())
	       << endl;
	}
    }
  else
    {
      h = _hnlsref[varName];
    }
  return h;
}

double
MERun::getVal( int table, int var, int i1, int i2 )
{

  if( table==ME::iLmfLaserPrim || table==ME::iLmfTestPulsePrim || table==ME::iLmfLEDPrim )
    {
      int ix = i1;
      int iy = i2;
      TH2* h_ = APDHist( var );
      if( h_==0 ) return 0;
      int binx = h_->GetXaxis()->FindBin( ix+0.5 );
      int biny = h_->GetYaxis()->FindBin( iy+0.5 );
      //int binx = h_->GetXaxis()->FindBin( ix );
      //      int biny = h_->GetYaxis()->FindBin( iy );
      double val =  (double) h_->GetCellContent( binx, biny );
      return val;  
    }
  else if( table==ME::iLmfLaserPnPrim || table==ME::iLmfTestPulsePnPrim|| table==ME::iLmfLEDPnPrim )
    {

      int ilm = i1;
      int ipn = i2;

      assert( ipn==0 || ipn==1 );

      // get the PN identifier

      pair<int,int> p_;
      int idcc=_header.dcc;
      int iside=_header.side;
      int ilmr=ME::lmr(idcc, iside);
      bool isBarrel=ME::isBarrel(ilmr);
      
      if( isBarrel ){
	assert( ilm>=1 && ilm<=9 );
	p_= MEEBGeom::pn(ilm, idcc);
      }else {
	int dee=MEEEGeom::dee( ilmr );
	p_= MEEEGeom::pn( dee, ilm );
	p_.first=p_.first+100;// 101 replaced by 100 (JM 24/04/10)
	p_.second=p_.second+200;// 201 replaced by 200 (JM 24/04/10)
      }
      
      int pnid = (ipn==0)? p_.first : p_.second; 
     
      TTree* tree_ = PNTable();
      int nentries = tree_->GetEntriesFast();
      int jj;
      for( jj=0; jj<nentries; jj++ )
	{
	  tree_->LoadTree( jj );
	  tree_->GetEntry( jj );
	  int pnid_ = pn_i["LOGIC_ID"]%10000;
	  if( pnid_==pnid ) break;
	}
      if( jj==nentries ) return 0;
      TString varName;
      if( table==ME::iLmfLaserPnPrim )
	varName=ME::PNPrimVar[var];
      else if( table==ME::iLmfTestPulsePnPrim )
	varName=ME::TPPNPrimVar[var];
      double val = (double) pn_d[varName];
      
      return val;
    }
  else if( table==ME::iLmfLaserPulse )
    {
      TTree* tree_ = MTQTable();
      int jj=_header.side;
      assert( jj==0 || jj==1 );
      tree_->LoadTree( 0 );
      tree_->GetEntry( 0 );
      assert( var>=0 && var<ME::iSizeMTQ );
      TString varName = ME::MTQPrimVar[var];
      double val = (double) mtq_d[varName];

      return val;
    }
  else if( table==ME::iLmfNLS)
    {
      bool doesExist=true;
      int ix = i1;
      int iy = i2;
      TH2* h_ = NLSHist( var , doesExist );
      
      if( h_==0 || doesExist==false ) return -99;

      int binx = h_->GetXaxis()->FindBin( ix+0.5 );
      int biny = h_->GetYaxis()->FindBin( iy+0.5 );
      // int binx = h_->GetXaxis()->FindBin( ix );
      // int biny = h_->GetYaxis()->FindBin( iy );
      double val =  (double) h_->GetCellContent( binx, biny );
      return val;  
    }
  else if( table==ME::iLmfNLSRef )
    {
      bool doesExist=true;
      int ix = i1;
      int iy = i2;
      TH2* h_ = NLSRefHist( var , doesExist );
     
      //      cout<< "h name "<<h_->GetName()<<endl;
      if( h_==0 || doesExist==false ) return -99;
      int binx = h_->GetXaxis()->FindBin( ix+0.5 );
      int biny = h_->GetYaxis()->FindBin( iy+0.5 );
      double val =  (double) h_->GetCellContent( binx, biny );
      //      cout<< "val "<< int(val)<<" bins: "<<binx<<" "<<biny<< " coord: "<<ix<<" "<<iy<<endl;
      return val;  
    }


  return 0;
}

bool
MERun::operator==( const MERun& o ) const
{
  bool out;
  out = ( _header.run  == o._header.run  )
    &&  ( _header.lb   == o._header.lb   )
    &&  ( _header.side == o._header.side )
    &&  ( _settings.wavelength == o._settings.wavelength );
  return out;
}

void
MERun::print( ostream& o ) const
{
  o << "\tMERUN:"<< endl;
  o << "\tRun    \t=\t" << _header.run    << endl;
  o << "\tLB     \t=\t" << _header.lb     << endl;
  o << "\tDCC    \t=\t" << _header.dcc    << endl;
  o << "\tSide   \t=\t" << _header.side   << endl;
  o << "\tEvents \t=\t" << _header.events << endl;
  //  o << hex;
  o << "\tTSBegin\t=\t" << _header.ts_beg << endl;
  o << "\tTSEnd  \t=\t" << _header.ts_end << endl;
  //  o << "\tTSLow  \t=\t" << time_low()     << endl;  
  o << "\tTime   \t=\t" << time()    << endl;  
  //  o << dec;
  o << "\tDt(sec)\t=\t" << _header.ts_end-_header.ts_beg << endl;
  o << "\tColor  \t=\t" << _settings.wavelength << endl;
  o << "\tPower  \t=\t" << _settings.power      << endl;
  o << "\tFilter \t=\t" << _settings.filter     << endl;
  o << "\tDelay  \t=\t" << _settings.delay      << endl;
  o << "\tMGPA   \t=\t" << _settings.mgpagain   << endl;
  o << "\tMEM    \t=\t" << _settings.memgain    << endl;
  o << "---> ROOT file name for primitives " << _fname << endl;
  if( _settings.type== ME::iLaser ) o << "---> ROOT file name for NLS       " << _fnlsname << endl;
}

//unsigned int
//MERun::time_low() const
//{
//  return ME::time_low( _header.ts_beg );
//}

// time in seconds
ME::Time
MERun::time() const
{
  return ME::time_high( _header.ts_beg );
}
