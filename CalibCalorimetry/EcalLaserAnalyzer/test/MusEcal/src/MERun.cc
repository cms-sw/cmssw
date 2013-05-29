#define MERun_cxx
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cassert>
using namespace std;

#include "MERun.hh"
#include "../../interface/MELaserPrim.h"
#include "../../interface/MEGeom.h"

ClassImp( MERun )

MERun::MERun( ME::Header header, ME::Settings settings, TString fname )
: _header( header ), _settings( settings ), _fname(fname), _file(0), pn_t(0), mtq_t(0)
{
  _type = _settings.type;
  _color = _settings.wavelength;

  //  assert( _color>=0 && _color<ME::iSizeC );  
  //  TTree* tree_ = PNTable();
  //   int nentries = tree_->GetEntriesFast();
  //   for( int jj=0; jj<nentries; jj++ )
  //     {
  //       //      int jj=(ilm-1)*2+ipn;
  //       tree_->LoadTree( jj );
  //       tree_->GetEntry( jj );
  //       //      assert( var>=0 && var<ME::iSizePN );
  //       cout << jj << " " << pn_i["LOGIC_ID"] << endl;
  //     }
  // 
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

  TH2* h(0);

  TFile* f = laserPrimFile();
  if( f==0 ) return h;
  
  TString histName = MELaserPrim::lmfLaserName( table, _type, _settings.wavelength );
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

  TString tableName = MELaserPrim::lmfLaserName( table, _type, _settings.wavelength );
  // don't cache the pn_t pointer, it's dangerous
  pn_i.clear();
  pn_d.clear();
  pn_t = (TTree*) f->Get(tableName);
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
  TString tableName = MELaserPrim::lmfLaserName( table, _type, _settings.wavelength );
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

float
MERun::getVal( int table, int var, int i1, int i2 )
{
  if( table==ME::iLmfLaserPrim || table==ME::iLmfTestPulsePrim )
    {
      int ix = i1;
      int iy = i2;
      TH2* h_ = APDHist( var );
      if( h_==0 ) return 0;
      int binx = h_->GetXaxis()->FindBin( ix+0.5 );
      int biny = h_->GetYaxis()->FindBin( iy+0.5 );
      //int binx = h_->GetXaxis()->FindBin( ix );
      //      int biny = h_->GetYaxis()->FindBin( iy );
      float val =  (float) h_->GetCellContent( binx, biny );
      return val;  
    }
  else if( table==ME::iLmfLaserPnPrim || table==ME::iLmfTestPulsePnPrim )
    {
      //      assert( var>=0 && var<ME::iSizePN );
      int ilm = i1;
      //      assert( ilm>=1 && ilm<=9 );
      if( !( ilm>=1 && ilm<=9 ) )
	{
	  //	  cout << "wrong module number for barrel! ilm=" << ilm << endl;  
	  return 0.; // SUPER FIXME !!!
	}
      int ipn = i2;
      assert( ipn==0 || ipn==1 );
      // get the PN identifier
      pair<int,int> p_ = MEEBGeom::pn(ilm);
      int pnid = (ipn==0)? p_.first : p_.second; 
      TTree* tree_ = PNTable();
      int nentries = tree_->GetEntriesFast();
      int jj;
      for( jj=0; jj<nentries; jj++ )
	{
	  tree_->LoadTree( jj );
	  tree_->GetEntry( jj );
	  int pnid_ = pn_i["LOGIC_ID"]%10000;
//  	  cout << "ilm/ipn/pnid/jj/pn_i/pnid_/pnid " 
// 	       << ilm << "/" 
// 	       << ipn << "/" 
// 	       << pnid << "/" 
// 	       << jj << "/" 
// 	       << pn_i["LOGIC_ID"] << "/" << pnid_ << endl;
	  if( pnid_==pnid ) break;
	}
      if( jj==nentries ) return 0;
      TString varName;
      if( table==ME::iLmfLaserPnPrim )
	varName=ME::PNPrimVar[var];
      else if( table==ME::iLmfTestPulsePnPrim )
	varName=ME::TPPNPrimVar[var];
      float val = (float) pn_d[varName];
      //      cout << pn_i["LOGIC_ID"] << " " << ilm << " " << ipn 
      //	   << " " << varName << "=" << val << endl;
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
      float val = (float) mtq_d[varName];

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
  o << "---> ROOT file name " << _fname << endl;
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
