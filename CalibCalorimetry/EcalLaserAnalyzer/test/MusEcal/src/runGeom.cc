#include <iostream>
#include <math.h>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <vector>
#include <map>

#include <assert.h>

//#include <istrstream>

#include <TChain.h>
#include <TFile.h>
#include <TObjString.h>
#include <TString.h>
#include <TStopwatch.h>
#include <TGraph.h>

using namespace std;

#include "../../interface/ME.h"
#include "MECanvasHolder.hh"
#include "../../interface/MEGeom.h"

void writeEBGeom()
{
  const int nsupermodules   =  36;

  TFile* test= new TFile( "ebgeom.root", "RECREATE" );

  //  list< pair<int,int> > l;

  TH2* eb_loc      = new TH2F( "eb_loc", "eb_loc", 95, -5.5, 89.5, 24, -2.5, 21.5 );
  MECanvasHolder::setHistoStyle( eb_loc );

  TH2* eb          = new TH2F( "eb", "eb", 190, -95.5, 94.5, 380, -9.5, 370.5 );
  MECanvasHolder::setHistoStyle( eb );

  TH2* eb_sm       = (TH2*) eb->Clone( "eb_sm"       );
  TH2* eb_dcc      = (TH2*) eb->Clone( "eb_dcc"      );
  TH2* eb_side     = (TH2*) eb->Clone( "eb_side"     );
  TH2* eb_lmmod    = (TH2*) eb->Clone( "eb_lmmod"    );
  TH2* eb_lmr      = (TH2*) eb->Clone( "eb_lmr"      );
  TH2* eb_tt       = (TH2*) eb->Clone( "eb_tt"       );
  TH2* eb_cr_in_sm = (TH2*) eb->Clone( "eb_cr_in_sm" );

  TH2* eb_loc_side     = (TH2*) eb_loc->Clone( "eb_loc_side"     );
  TH2* eb_loc_lmmod    = (TH2*) eb_loc->Clone( "eb_loc_lmmod"    );
  TH2* eb_loc_cr       = (TH2*) eb_loc->Clone( "eb_loc_cr"       );
  TH2* eb_loc_elecr    = (TH2*) eb_loc->Clone( "eb_loc_elecr"    );
  TH2* eb_loc_hv       = (TH2*) eb_loc->Clone( "eb_loc_hv"       );
  TH2* eb_loc_lv       = (TH2*) eb_loc->Clone( "eb_loc_lv"       );
  TH2* eb_loc_tt       = (TH2*) eb_loc->Clone( "eb_loc_tt"       );

  for( int ii=1; ii<=nsupermodules; ii++ )
    {
      TGraph* g_ = MEEBGeom::getGraphBoundary( MEEBGeom::iSuperModule, ii, true );
      assert( g_!=0 );
      TString gname = "SuperModule_"; gname += ii;
      g_->SetName( gname );
      g_->Write();
    }
 
  {
    TGraph* g_ = MEEBGeom::getGraphBoundary( MEEBGeom::iSuperModule, 1, false );
    assert( g_!=0 );
    TString gname = "SuperModule";
    g_->SetName( gname );
    g_->Write();
  }

  for( int ii=1; ii<=72; ii++ )
    {      
      TGraph* g_ = MEEBGeom::getGraphBoundary( MEEBGeom::iLMRegion, ii, true );
      assert( g_!=0 );
      TString gname = "MonitoringRegion_"; gname += ii;
      g_->SetName( gname );
      g_->Write();
    }

  for( int iside=0; iside<2; iside++ )
    {
      TGraph* g_ = MEEBGeom::getGraphBoundary( MEEBGeom::iLMRegion, iside, false );
      assert( g_!=0 );
      TString gname = "Side_";
      gname += iside;
      g_->SetName( gname );
      g_->Write();
    }

  for( int ieta=-85; ieta<=85; ieta++ )
    {
      if( ieta==0 ) continue;
      for( int iphi=1; iphi<=360; iphi++ )
	{	  

	  int ism = MEEBGeom::sm( ieta, iphi );
	  assert( ism>0 );

	  int idcc = MEEBGeom::dcc( ieta, iphi );
	  assert( idcc>0 );

	  int iside = MEEBGeom::side( ieta, iphi );
	  int ilmr  = MEEBGeom::lmr( ieta, iphi );
	  int itt   = MEEBGeom::tt( ieta, iphi );
	  int icr_in_sm   = MEEBGeom::crystal( ieta, iphi );

	  int ilmmod = MEEBGeom::lmmod( ieta, iphi );
	  // int ilmmod = 1;
	  assert( ilmmod>0 );

	  eb_sm     -> Fill( ieta, iphi, ism    );
	  eb_dcc    -> Fill( ieta, iphi, idcc   );
	  eb_side   -> Fill( ieta, iphi, iside  );
	  eb_lmr    -> Fill( ieta, iphi, ilmr   );
	  eb_lmmod  -> Fill( ieta, iphi, ilmmod );
	  eb_tt     -> Fill( ieta, iphi, itt    );
	  eb_cr_in_sm  -> Fill( ieta, iphi, icr_in_sm    );

	}
    }

  for( int ix=0; ix<=84; ix++ )
    {
      for( int iy=0; iy<=19; iy++ )
	{	  
	  int icr    = MEEBGeom::crystal_channel( ix, iy );
	  int ielecr = MEEBGeom::electronic_channel( ix, iy );

	  int iX = ix/5;
	  int iY = iy/5;	  
	  int ihv   = MEEBGeom::hv_channel( iX, iY );
	  int ilv   = MEEBGeom::lv_channel( iX, iY );
	  int itt   = MEEBGeom::tt_channel( iX, iY );
	  int ilmmod = MEEBGeom::lm_channel( iX, iY );
	  int iside = (ilmmod%2==0)?1:0;
	  assert( ilmmod>0 );

	  eb_loc_side   -> Fill( ix, iy, iside+0.1  );
	  eb_loc_lmmod  -> Fill( ix, iy, ilmmod );
	  eb_loc_tt     -> Fill( ix, iy, itt    );
	  eb_loc_cr     -> Fill( ix, iy, icr    );
	  eb_loc_elecr  -> Fill( ix, iy, ielecr );
	  eb_loc_hv     -> Fill( ix, iy, ihv    );
	  eb_loc_lv     -> Fill( ix, iy, ilv    );

	}
    }

  eb->Write();
  eb_sm->Write();
  eb_dcc->Write();
  eb_side->Write();
  eb_lmmod->Write();
  eb_lmr->Write();
  eb_tt->Write();
  eb_cr_in_sm->Write();

  eb_loc->Write();
  eb_loc_side->Write();
  eb_loc_lmmod->Write();
  eb_loc_tt->Write();
  eb_loc_cr->Write();
  eb_loc_elecr->Write();
  eb_loc_hv->Write();
  eb_loc_lv->Write();

  test->Close();

}

void writeEEGeom( int iz )
{
  //  int iz=-1;
  const int nsectors   =  9;
  const int nlmregions = 10;
  const int nlmmodules = 19;
  const int nquadrants =  4;

  TString fname = "eegeom_"; 
  fname += (iz>0)?2:1;
  fname += ".root";
  TFile* test= new TFile( fname, "RECREATE" );

  //  list< pair<int,int> > l;

  TH2* eem          = new TH2F( "eem", "eem", 110,-4.5,105.5,110,-4.5,105.5 );
  MECanvasHolder::setHistoStyle( eem );
  TH2* eem_sect     = (TH2*) eem->Clone( "eem_sect" );
  TH2* eem_quad     = (TH2*) eem->Clone( "eem_quad" );
  TH2* eem_sc       = (TH2*) eem->Clone( "eem_sc" );
  TH2* eem_lmmod    = (TH2*) eem->Clone( "eem_lmmod" );
  TH2* eem_dcc      = (TH2*) eem->Clone( "eem_dcc" );
  TH2* eem_lmr      = (TH2*) eem->Clone( "eem_lmr" );
  TH2* eem_cr       = (TH2*) eem->Clone( "eem_cr" );
  TH2* eem_cr_in_sc = (TH2*) eem->Clone( "eem_cr_in_sc" );

  TH2* eem_loc[nsectors+1];
  TH2* eem_loc_lmmod[nsectors+1];
  TH2* eem_loc_cr[nsectors+1];
  TH2* eem_loc_sc[nsectors+1];

  // loop on sectors
  for( int ii=1; ii<=nsectors; ii++ )
    {
      TGraph* g_ = MEEEGeom::getGraphBoundary( MEEEGeom::iSector, ii );
      assert( g_!=0 );
      TString gname = "Sector_"; gname += ii;
      g_->SetName( gname );
      g_->Write();
      
      // determine the histogram boundaries
      double xmin=101;
      double xmax=-1;
      double ymin=101;
      double ymax=-1;
      for( int jj=0; jj<g_->GetN(); jj++ )
	{
	  double x,y;
	  g_->GetPoint( jj, x, y );
	  if( x<xmin ) xmin=x;
	  if( x>xmax ) xmax=x;
	  if( y<ymin ) ymin=y;
	  if( y>ymax ) ymax=y;
	}
      xmin -=5;
      xmax +=5;
      ymin -=5;
      ymax +=5;

      TString hname0("eem_S");
      hname0 += ii;
      eem_loc[ii] = new TH2F( hname0, hname0, 
			      (int)(xmax-xmin), (float) xmin, (float) xmax, 
			      (int)(ymax-ymin), (float) ymin, (float) ymax );
      MECanvasHolder::setHistoStyle( eem_loc[ii] );
      
      TString hname;
      hname = hname0; hname += "_lmmod";
      eem_loc_lmmod[ii] = (TH2*) eem_loc[ii]->Clone( hname );
      hname = hname0; hname += "_sc";
      eem_loc_sc[ii] = (TH2*) eem_loc[ii]->Clone( hname );
      hname = hname0; hname += "_cr";
      eem_loc_cr[ii] = (TH2*) eem_loc[ii]->Clone( hname );

    }

  for( int ii=1; ii<=nlmregions; ii++ )
    {
      int lmr_ = ii;
      if( iz==-1 ) lmr_+=82;
      else if( iz==+1 ) lmr_+=72;
      else abort();
      TGraph* g_ = MEEEGeom::getGraphBoundary( MEEEGeom::iLMRegion, lmr_, iz );
      assert( g_!=0 );
      TString gname = "LMRegion_"; gname += ii;
      g_->SetName( gname );
      g_->Write();
    }

  for( int ii=1; ii<=nquadrants; ii++ )
    {
      TGraph* g_ = MEEEGeom::getGraphBoundary( MEEEGeom::iQuadrant, ii );
      assert( g_!=0 );
      TString gname = "Quadrant_"; gname += ii;
      g_->SetName( gname );
      g_->Write();
    }

  for( int xside=1; xside<=2; xside++ )
    {
      for( int ilmmod=1; ilmmod<=nlmmodules; ilmmod++ )
	{
	  TGraph* g_ = MEEEGeom::getGraphBoundary( MEEEGeom::iLMModule, ilmmod, iz, xside );
	  if( g_==0 ) continue;
	  TString gname = "LMModule_"; gname += ilmmod; gname += "_"; gname += xside;
	  g_->SetName( gname );
	  g_->Write();
	}
    }

  for( int ix=1; ix<=100; ix++ )
    {
      for( int iy=1; iy<=100; iy++ )
	{	  
	  int icr = MEEEGeom::crystal( ix, iy );
	  bool ok = icr>0;
	  if( !ok ) 
	    continue;

	  int icr_in_sc = MEEEGeom::crystal_in_sc( ix, iy );
	  
	  int iX = (ix-1)/5+1;
	  int iY = (iy-1)/5+1;
	  
	  int isect = MEEEGeom::sector( iX, iY );
	  assert( isect>=1 && isect<=nsectors );
	  
	  int iquad = MEEEGeom::quadrant( iX, iY );
	  assert( iquad>0 );
	  
	  int isc   = MEEEGeom::sc( iX, iY );
	  assert( isc>0 );
	  
	  int ilmmod = MEEEGeom::lmmod( iX, iY );
	  assert( ilmmod>0 );
	  
	  int idcc= MEEEGeom::dcc( iX, iY, iz );
	  assert( idcc>0 );
	  
	  int ilmr= MEEEGeom::lmr( iX, iY, iz );
	  assert( ilmr>0 );

	  eem_sect  -> Fill( ix, iy, isect  );
	  eem_quad  -> Fill( ix, iy, iquad  );
	  eem_sc    -> Fill( ix, iy, isc    );
	  eem_lmmod -> Fill( ix, iy, ilmmod );
	  eem_dcc   -> Fill( ix, iy, idcc   );
	  eem_lmr   -> Fill( ix, iy, ilmr   );
	  eem_cr    -> Fill( ix, iy, icr    );
	  eem_cr_in_sc   -> Fill( ix, iy, icr_in_sc  );

	  eem_loc_lmmod[isect] -> Fill( ix, iy, ilmmod );
	  eem_loc_sc[isect]    -> Fill( ix, iy, isc );
	  eem_loc_cr[isect]    -> Fill( ix, iy, icr );
	}
    }

  eem->Write();
  eem_sect->Write();
  eem_quad->Write();
  eem_sc->Write();
  eem_lmmod->Write();
  eem_dcc->Write();
  eem_lmr->Write();
  eem_cr->Write();
  eem_cr_in_sc->Write();

  for( int isect=1; isect<=nsectors; isect++ )
    {
      eem_loc[isect]->Write();
      eem_loc_lmmod[isect]->Write();
      eem_loc_sc[isect]->Write();
      eem_loc_cr[isect]->Write();
    }

  test->Close();

}

void writeEEGeom()
{
  //  int iz=-1;
  const int nsectors   =  9;
  const int nlmregions = 10;
  const int nlmmodules = 19;
  const int nquadrants =  4;

  TString fname = "eegeom"; 
  fname += ".root";
  TFile* test= new TFile( fname, "RECREATE" );

  //  list< pair<int,int> > l;

  TH2* ee          = new TH2F( "ee", "ee", 110,-4.5,105.5,211,-105.5,105.5 );
  MECanvasHolder::setHistoStyle( ee );
  TH2* ee_sect     = (TH2*) ee->Clone( "ee_sect" );
  TH2* ee_quad     = (TH2*) ee->Clone( "ee_quad" );
  TH2* ee_sc       = (TH2*) ee->Clone( "ee_sc" );
  TH2* ee_lmmod    = (TH2*) ee->Clone( "ee_lmmod" );
  TH2* ee_dcc      = (TH2*) ee->Clone( "ee_dcc" );
  TH2* ee_lmr      = (TH2*) ee->Clone( "ee_lmr" );
  TH2* ee_cr       = (TH2*) ee->Clone( "ee_cr" );
  TH2* ee_cr_in_sc = (TH2*) ee->Clone( "ee_cr_in_sc" );

  TH2* ee_loc[2*nsectors+1];
  TH2* ee_loc_lmmod[2*nsectors+1];
  TH2* ee_loc_cr[2*nsectors+1];
  TH2* ee_loc_sc[2*nsectors+1];

  for( int kk=0; kk<2; kk++ )
    {
      int iz = 1 - 2*kk;
      // kk=0 EEP, kk=1 EEM
      // loop on sectors
      for( int ii=1; ii<=nsectors; ii++ )
	{
	  int isect = ii + kk*nsectors;
	  TGraph* g_ = MEEEGeom::getGraphBoundary( MEEEGeom::iSector, ii );
	  assert( g_!=0 );
	  for( int jj=0; jj<g_->GetN(); jj++ )
	    {
	      double x,y;
	      g_->GetPoint( jj, x, y );
	      g_->SetPoint( jj, x, iz*y );
	    }
	  TString gname = "Sector_"; gname += isect;
	  g_->SetName( gname );
	  g_->Write();
      
	  // determine the histogram boundaries
	  double xmin=101;
	  double xmax=-1;
	  double ymin=101;
	  double ymax=-101;
	  for( int jj=0; jj<g_->GetN(); jj++ )
	    {
	      double x,y;
	      g_->GetPoint( jj, x, y );
	      if( x<xmin ) xmin=x;
	      if( x>xmax ) xmax=x;
	      if( y<ymin ) ymin=y;
	      if( y>ymax ) ymax=y;
	    }
	  xmin -=5;
	  xmax +=5;
	  ymin -=5;
	  ymax +=5;

	  TString hname0("ee_S");
	  hname0 += isect;
	  ee_loc[isect] = new TH2F( hname0, hname0, 
				     (int)(xmax-xmin), (float) xmin, (float) xmax, 
				     (int)(ymax-ymin), (float) ymin, (float) ymax );
	  MECanvasHolder::setHistoStyle( ee_loc[isect] );
      
	  TString hname;
	  hname = hname0; hname += "_lmmod";
	  ee_loc_lmmod[isect] = (TH2*) ee_loc[isect]->Clone( hname );
	  hname = hname0; hname += "_sc";
	  ee_loc_sc[isect] = (TH2*) ee_loc[isect]->Clone( hname );
	  hname = hname0; hname += "_cr";
	  ee_loc_cr[isect] = (TH2*) ee_loc[isect]->Clone( hname );

	}

      //      if( 1 ) continue;



      for( int ii=1; ii<=nlmregions; ii++ )
	{
	  int lmr_ = 72+ii+kk*nlmregions;
	  TGraph* g_ = MEEEGeom::getGraphBoundary( MEEEGeom::iLMRegion, lmr_, iz );
	  assert( g_!=0 );
	  for( int jj=0; jj<g_->GetN(); jj++ )
	    {
	      double x,y;
	      g_->GetPoint( jj, x, y );
	      g_->SetPoint( jj, x, iz*y );
	    }
	  TString gname = "LMRegion_"; gname += lmr_;
	  g_->SetName( gname );
	  g_->Write();
	}

      for( int ii=1; ii<=nquadrants; ii++ )
	{
	  int quad_ = ii+kk*nquadrants;
	  TGraph* g_ = MEEEGeom::getGraphBoundary( MEEEGeom::iQuadrant, ii );
	  assert( g_!=0 );
	  for( int jj=0; jj<g_->GetN(); jj++ )
	    {
	      double x,y;
	      g_->GetPoint( jj, x, y );
	      g_->SetPoint( jj, x, iz*y );
	    }
	  TString gname = "Quadrant_"; gname += quad_;
	  g_->SetName( gname );
	  g_->Write();
	}

      for( int xside=1; xside<=2; xside++ )
	{
	  for( int ilmmod=1; ilmmod<=nlmmodules; ilmmod++ )
	    {
	      int mod_ = ilmmod + kk*nlmmodules;
	      TGraph* g_ = MEEEGeom::getGraphBoundary( MEEEGeom::iLMModule, ilmmod, iz, xside );
	      if( g_==0 ) continue;
	      for( int jj=0; jj<g_->GetN(); jj++ )
		{
		  double x,y;
		  g_->GetPoint( jj, x, y );
		  g_->SetPoint( jj, x, iz*y );
		}
	      TString gname = "LMModule_"; gname += mod_; gname += "_"; gname += xside;
	      g_->SetName( gname );
	      g_->Write();
	    }
	}

      for( int ix=1; ix<=100; ix++ )
	{
	  for( int iy=1; iy<=100; iy++ )
	    {	  
	      int icr = MEEEGeom::crystal( ix, iy );
	      bool ok = icr>0;
	      if( !ok ) 
		continue;

	      int icr_in_sc = MEEEGeom::crystal_in_sc( ix, iy );
	  
	      int iX = (ix-1)/5+1;
	      int iY = (iy-1)/5+1;
	  
	      int isect = MEEEGeom::sector( iX, iY );
	      assert( isect>=1 && isect<=nsectors );
	      int sect_ = isect+kk*nsectors;
	  
	      int iquad = MEEEGeom::quadrant( iX, iY );
	      assert( iquad>0 );
	  
	      int isc   = MEEEGeom::sc( iX, iY );
	      assert( isc>0 );
	  
	      int ilmmod = MEEEGeom::lmmod( iX, iY );
	      assert( ilmmod>0 );
	  
	      int idcc= MEEEGeom::dcc( iX, iY, iz );
	      assert( idcc>0 );
	  
	      int ilmr= MEEEGeom::lmr( iX, iY, iz );
	      assert( ilmr>0 );

	      ee_sect  -> Fill( ix, iz*iy, isect  );
	      ee_quad  -> Fill( ix, iz*iy, iquad  );
	      ee_sc    -> Fill( ix, iz*iy, isc    );
	      ee_lmmod -> Fill( ix, iz*iy, ilmmod );
	      ee_dcc   -> Fill( ix, iz*iy, idcc   );
	      ee_lmr   -> Fill( ix, iz*iy, ilmr   );
	      ee_cr    -> Fill( ix, iz*iy, icr    );
	      ee_cr_in_sc   -> Fill( ix, iz*iy, icr_in_sc  );

	      ee_loc_lmmod[sect_] -> Fill( ix, iz*iy, ilmmod );
	      ee_loc_sc[sect_]    -> Fill( ix, iz*iy, isc );
	      ee_loc_cr[sect_]    -> Fill( ix, iz*iy, icr );
	    }
	}
    }

  ee->Write();
  ee_sect->Write();
  ee_quad->Write();
  ee_sc->Write();
  ee_lmmod->Write();
  ee_dcc->Write();
  ee_lmr->Write();
  ee_cr->Write();
  ee_cr_in_sc->Write();

  for( int isect=1; isect<=2*nsectors; isect++ )
    {
      ee_loc[isect]->Write();
      ee_loc_lmmod[isect]->Write();
      ee_loc_sc[isect]->Write();
      ee_loc_cr[isect]->Write();
    }

  test->Close();

}

void writeGeom()
{
  TFile* test= new TFile( "ecalgeom.root", "RECREATE" );

  //  list< pair<int,int> > l;

  TH2* ecal_sm            = MEGeom::getGlobalHist( "Sectors" );
  TH2* ecal_lmr           = MEGeom::getGlobalHist( "MonitoringRegions" );

  MECanvasHolder::setHistoStyle( ecal_sm );
  MECanvasHolder::setHistoStyle( ecal_lmr );

  for( int ieta=-85; ieta<=85; ieta++ )
    {
      if( ieta==0 ) continue;
      for( int iphi=1; iphi<=360; iphi++ )
	{	  

	  int ism = MEEBGeom::sm( ieta, iphi );
	  assert( ism>0 );

	  int ilmr = MEEBGeom::lmr( ieta, iphi );
	  assert( ilmr>0 );

	  MEGeom::setBinGlobalHist( ecal_sm,  ieta, iphi, 0, (float)ism  );
	  MEGeom::setBinGlobalHist( ecal_lmr, ieta, iphi, 0, (float)ilmr );
	}
    }

  for( int iz=-1; iz<=1; iz+=2 )
    { 
      for( int ix=1; ix<=100; ix++ )
	{
	  for( int iy=1; iy<=100; iy++ )
	    {	  
	      int icr = MEEEGeom::crystal( ix, iy );
	      bool ok = icr>0;
	      if( !ok ) 
		continue;
	      
	      int iX = (ix-1)/5+1;
	      int iY = (iy-1)/5+1;
	      
	      int isect = MEEEGeom::sector( iX, iY );
	      int ilmr  = MEEEGeom::lmr( iX, iY, iz );
	      
	      MEGeom::setBinGlobalHist( ecal_sm,  ix, iy, iz, (float)isect ); 
	      MEGeom::setBinGlobalHist( ecal_lmr, ix, iy, iz, (float)ilmr  ); 

	    }
	}
    }

  ecal_sm->Write();
  ecal_lmr->Write();
  test->Close();
}

int main(int argc, char **argv)
{

  //  ofstream o("eenum.txt");

  //
  // Test that the geometry corresponds to that of EEDetId
  //
  //   for( int iz=+1; iz>=-1; iz-=2 )
  //     {
  //       for( int ix=1; ix<=100; ix++ )
  // 	{
  // 	  for( int iy=1; iy<=100; iy++ )
  // 	    {
  // 	      //	      bool ok = EEDetId::validDetId( ix, iy, iz ); 
  // 	      int icr = MEEEGeom::crystal( ix, iy );
  // 	      bool ok = icr>=0;
  // 	      if( !ok ) 
  // 		{
  // 		  //		  assert( icr<0 );
  // 		  continue;
  // 		}
  // 	      //	      EEDetId id( ix, iy, iz );
  // 	      //	      EEDetId id_sc( id.isc(), id.ic(), iz, 1 );

  // 	      int icr_in_sc = MEEEGeom::crystal_in_sc( ix, iy );
		  
  // 	      int iX = (ix-1)/5+1;
  // 	      int iY = (iy-1)/5+1;

  // 	      int isect = MEEEGeom::sector( iX, iY );
  // 	      assert( isect!=0 );
  // 	      if( isect<0 ) continue;

  // 	      int iquad = MEEEGeom::quadrant( iX, iY );
  // 	      //	      int idee  = MEEEGeom::dee( iX, iY, iz );
  // 	      int isc   = MEEEGeom::sc( iX, iY );
  // 	      assert( isc>0 );

  // 	      int ilmmod = MEEEGeom::lmmod( iX, iY );
  // 	      assert( ilmmod>0 );

  // 	      int idcc= MEEEGeom::dcc( iX, iY, iz );
  // 	      assert( idcc>0 );

  // 	      int ilmr= MEEEGeom::lmr( iX, iY, iz );
  // 	      assert( ilmr>0 );

  // 	      o << "ix=" << id.ix();
  // 	      o << "\tiy=" << id.iy();
  // 	      o << "\tiz=" << id.zside();
  // 	      o << "\tisc=" << id.isc();
  // 	      o << "\tic=" << id.ic();
  // 	      o << "\tiquad=" << id.iquadrant();
  // 	      o << endl;
  // 	      o << "ix=" << id_sc.ix();
  // 	      o << "\tiy=" << id_sc.iy();
  // 	      o << "\tiz=" << id_sc.zside();
  // 	      o << "\tisc=" << id_sc.isc();
  // 	      o << "\tic=" << id_sc.ic();
  // 	      o << "\tiquad=" << id_sc.iquadrant();
  // 	      o << endl;

  // 	      assert( id.ix()==id_sc.ix() );
  // 	      assert( id.iy()==id_sc.iy() );
  // 	      assert( id.zside()==id_sc.zside() );
  // 	      assert( id.isc()==id_sc.isc() );
  // 	      assert( id.ic()==id_sc.ic() );
  // 	      assert( id.iquadrant()==id_sc.iquadrant() );
  // 	      assert( ix==id.ix() );
  // 	      assert( iy==id.iy() );
  // 	      assert( iz==id.zside() );
  // 	      assert( isc==id.isc() );
  // 	      assert( icr_in_sc==id.ic() );
  // 	      assert( iquad==id.iquadrant() );

  // 	      o << " x=" << ix;
  // 	      o << "\t y=" << iy;
  // 	      o << "\t z=" << iz;
  // 	      o << "\t sc=" << isc;
  // 	      o << "\t c=" << icr_in_sc;		  
  // 	      o << "\tdee=" << idee;
  // 	      o << "\tquad=" << iquad;
  // 	      o << "\tsect=" << isect;
  // 	      o << "\tlmmod=" << ilmmod;
  // 	      o << "\tdcc=" << idcc;
  // 	      o << "\tlmr=" << ilmr;
  // 	      o << endl;
  //	    }
  //	}      
  //    }

  writeEEGeom( -1 );
  writeEBGeom();
  writeEEGeom( +1 );
  writeEEGeom();
  writeGeom();

  //   const int nlmregions      =  92;

  //   for ( int ii=1; ii<=nlmregions; ii++ )
  //     {
  //       TH2* h_ = MEGeom::getHist( ii, MEGeom::iSuperCrystal );
  //       assert( h_!=0 );
  //       h_->Print();
  //       cout << h_->GetTitle() << endl;
  //     }

  //   for( int ii=0; ii<ME::iSizeC; ii++ )
  //     {
  //       cout << ME::color[ii] << endl;
  //     }

  
  return(0);
}

