#include <cassert>
#include <iostream>
#include <stdlib.h>
#include <string>
using namespace std;

#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEGeom.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/ME.h"

#include <TCanvas.h>
#include <TGraph.h>
#include <TFile.h>
#include <TH2.h>

//TString MEGeom::granularity[MEGeom::iSizeG] = { 
//  "Ecal Region", "Ecal Sector", "Laser Monitoring Region", "Laser Monitoring Module", "Super-Crystal", "Crystal" 
//};

int MEGeom::_nbuf = 5;
int MEGeom::_nbinx = 2*( MEGeom::_nbuf + 100 + 85 ) + 1;
float MEGeom::_xmin = -( 0.5 + MEGeom::_nbuf + 100 + 85 );
float MEGeom::_xmax =    0.5 + MEGeom::_nbuf + 100 + 85  ;
int MEGeom::_nbiny = 2*( MEGeom::_nbuf + 180 );
float MEGeom::_ymin = 180.5 - ( MEGeom::_nbuf + 180 );  
float MEGeom::_ymax = 180.5 + ( MEGeom::_nbuf + 180 );
TH2* MEGeom::_h = new TH2F( "globalEcal", "Global representation of ECAL",
			    MEGeom::_nbinx, MEGeom::_xmin, MEGeom::_xmax, 
			    MEGeom::_nbiny, MEGeom::_ymin, MEGeom::_ymax );

//GHM ClassImp(MEGeom)


TH2* 
MEGeom::getHist( int ilmr, int unit )
{
  int ireg;
  int ism;
  int idcc; 
  int side;
  ME::regionAndSector( ilmr, ireg, ism, idcc, side );

  std::cout << "LM region=" << ilmr << " reg/sm/dcc/side " << ireg << "/" << ism << "/" << idcc << "/" << side << std::endl;

  TString hname = "LMR="; hname += ilmr; hname += " ";
  if( ireg==ME::iEEM || ireg==ME::iEEP )
    {
      hname += MEEEGeom::smName( ism );
    }
  else if( ireg==ME::iEBM || ireg==ME::iEBP )
    {
      hname += MEEBGeom::smName( ism );
    }
  else
    abort();

  hname += " ";
  hname += "DCC="; hname += idcc; hname += "/" ; hname += side;
  
  //
  // to produce these root files, run the runGeom executable
  //
  TH2* h_(0);
  TFile* rootfile(0);
  TString hn_;
  if( ireg==ME::iEBM || ireg==ME::iEBP )
    {
      rootfile = TFile::Open("ebgeom.root");
      assert( rootfile!=0 );
      hn_="eb_loc";
      switch ( unit )
	{
	case ME::iSector:             break;
	case ME::iLMRegion:           hn_ += "_side";   break;
	case ME::iLMModule:           hn_ += "_lmmod";  break;
	case ME::iSuperCrystal:       hn_ += "_tt";     break;
	case ME::iCrystal:            hn_ += "_cr";     break;
	case ME::iElectronicChannel:  hn_ += "_elecr";  break;
	case ME::iHVChannel:          hn_ += "_hv";     break;
	case ME::iLVChannel:          hn_ += "_lv";     break;
	}
    }
  else
    {
      int isect=ism;
      if( ireg==ME::iEEM )
	{
	  isect-=9;
	  rootfile = TFile::Open("eegeom_1.root");
	}
      if( ireg==ME::iEEP ) rootfile = TFile::Open("eegeom_2.root");
      assert( rootfile!=0 );
      hn_="eem_S"; hn_+= isect; 
      switch (unit)
	{
	case ME::iSector:             break;
	case ME::iLMRegion:   break;
	case ME::iLMModule:   hn_ += "_lmmod";  break;
	case ME::iSuperCrystal:       hn_ += "_sc";     break;
	case ME::iCrystal:            hn_ += "_cr";     break;
	case ME::iElectronicChannel:  hn_ += "_cr";  break;
	case ME::iHVChannel:          break;
	case ME::iLVChannel:          break;
	}
    }
  h_ = (TH2*) rootfile->Get( hn_ );
  h_->SetTitle( hname );
  h_->GetXaxis()->SetTitle("ix");
  h_->GetXaxis()->CenterTitle();
  h_->GetYaxis()->SetTitle("iy");
  h_->GetYaxis()->CenterTitle();
  return h_;
}

TGraph* 
MEGeom::getBoundary( int ilmr, int histtype )
{
  // for local pictures, only sector or monitoring region
  if( histtype!=ME::iSector && histtype!=ME::iLMRegion ) 
    histtype = ME::iSector;

  int ireg;
  int ism;
  int idcc; 
  int side;
  ME::regionAndSector( ilmr, ireg, ism, idcc, side );

  //
  // to produce these root files, run the runGeom executable
  //
  TGraph* g_(0);
  TFile* rootfile(0);
  TString gn_;
  if( ireg==ME::iEBM || ireg==ME::iEBP )
    {
      rootfile = TFile::Open("ebgeom.root");
      assert( rootfile!=0 );
      switch (histtype)
	{
	case ME::iSector:             gn_ = "SuperModule"; break;
	case ME::iLMRegion:   gn_ = "Side_"; gn_+=side; break;
	}
    }
  else
    {
      int isect=ism;
      if( ireg==ME::iEEM ) 
	{
	  isect-=9;
	  rootfile = TFile::Open("eegeom_1.root");
	}
      if( ireg==ME::iEEP ) rootfile = TFile::Open("eegeom_2.root");
      assert( rootfile!=0 );
      int lmr_= ilmr;
      if( ireg==ME::iEEP ) lmr_-=72;
      else if( ireg==ME::iEEM ) lmr_-=82;
      switch (histtype)
	{
	case ME::iSector:             gn_ = "Sector_"; gn_+=isect; break;
	case ME::iLMRegion:   gn_ = "LMRegion_"; gn_+=lmr_; break;
	}
    }
  g_ = (TGraph*) rootfile->Get( gn_ );
  return g_;
}

void
MEGeom::drawHist( int ilmr, int histtype, TCanvas* canv )
{

  TH2* h = getHist( ilmr, histtype );
  assert( h!=0 );
  TString tname = h->GetTitle();
  switch( histtype )
    {
    case ME::iSector: break;
    case ME::iLMRegion: tname += " Monitoring Regions"; break;
    case ME::iLMModule: tname += " Monitoring Modules"; break;
    case ME::iSuperCrystal: tname += " Super Crystals"; break;
    case ME::iCrystal: tname += " Crystals"; break;
    case ME::iElectronicChannel: tname += " Electronic Channels"; break;
    case ME::iHVChannel: tname += " HV Channels"; break;
    case ME::iLVChannel: tname += " LV Channels"; break;
    }
  
  if( canv==0 )
    {
      TString cname = tname;
      cname.ReplaceAll(" ","_");
      canv = new TCanvas( cname, cname, 10, 10, 500, 500 );
    }
  canv->SetTitle( tname );
  canv->cd();

  h->Draw("COLZ");

  TGraph* gsect = getBoundary( ilmr, ME::iSector );
  assert( gsect!=0 );
  gsect->SetLineWidth( 1 );
  gsect->Draw("LSame");
  TGraph* gside = getBoundary( ilmr, ME::iLMRegion );
  assert( gside!=0 );
  gside->SetLineWidth( 2 );
  gside->Draw("LSame");
}

TH2*
MEGeom::getGlobalHist( const char* name )
{
  TH2* h = (TH2*)_h->Clone( name );
  h->Reset();
  return h;
}

void
MEGeom::setBinGlobalHist( TH2* h, int ix, int iy, int iz, float val )
{
  // make sure it's a global hist
  TAxis* ax = h->GetXaxis();
  TAxis* ay = h->GetYaxis();
  assert( ax->GetNbins()==_nbinx && ay->GetNbins()==_nbiny );
  assert( ax->GetXmax()==_xmax && ax->GetXmin()==_xmin );
  assert( ay->GetXmax()==_ymax && ay->GetXmin()==_ymin );

  int ibinx(0);
  int ibiny(0);

  if( iz==0 )
    {
      int ieta = ix;
      int iphi = iy;
      assert( abs(ieta)>=1 && abs(ieta)<=85 );
      assert( iphi>=1 && iphi<=360 );
      
      ibinx = ax->FindBin( ieta );
      ibiny = ay->FindBin( iphi );
    }
  else if( iz==-1 )
    {
      assert( ix>=1 && ix<=100 && iy>=1 && iy<=100 );
      ibinx = (_nbuf+100+1)-ix;
      ibiny = (_nbuf+180+50+1)-iy;
    }
  else if( iz==1  )
    {
      assert( ix>=1 && ix<=100 && iy>=1 && iy<=100 );
      ibinx = _nbinx+1-((_nbuf+100+1)-ix);
      ibiny = (_nbuf+180+50+1)-iy;
    }
  h->SetBinContent( ibinx, ibiny, val );
}

// fixme !!!
void
MEGeom::drawGlobalBoundaries( int lineColor )
{
  TGraph* gr(0);
  for( int ism=1; ism<=36; ism++ )
    {
      gr = MEEBGeom::getGraphBoundary( MEEBGeom::iSuperModule, ism, true );
      gr->SetLineWidth( 2 );
      gr->SetLineColor( lineColor );
      gr->Draw("LSame");
    }
  for( int isec=1; isec<=9; isec++ )
    {
      gr = MEEEGeom::getGraphBoundary( MEEEGeom::iSector, isec );
      TGraph* grm = (TGraph*) gr->Clone();
      TGraph* grp = (TGraph*) gr->Clone();
      int n = gr->GetN();
      //      std::cout << std::endl;
      for( int ii=0; ii<n; ii++ )
	{
	  double x_, y_;
	  gr->GetPoint( ii, x_, y_ );
	  //	  std::cout << isec << " " << ii << " x=" << x_ << " y=" << y_ << std::endl;

	  double xx_, yy_;

	  xx_ = -85-x_;
	  yy_ = 231-y_;
	  //	  std::cout << isec << " " << 0 << " x=" << xx_ << " y=" << yy_ << std::endl;

	  grm->SetPoint( ii, xx_, yy_ );

	  xx_ =  85+x_;
	  yy_ =  231-y_;
	  //	  std::cout << isec << " " << 1 << " x=" << xx_ << " y=" << yy_ << std::endl;
	  grp->SetPoint( ii, xx_, yy_ );
	}
      grm->SetLineColor( lineColor );
      grm->SetLineWidth( 2 );
      grm->Draw("LSame");
      grp->SetLineColor( lineColor );
      grp->SetLineWidth( 2 );
      grp->Draw("LSame");
    }
}

