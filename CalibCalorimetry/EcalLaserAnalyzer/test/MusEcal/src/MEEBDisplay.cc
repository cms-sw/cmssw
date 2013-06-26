#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <string>
using namespace std;

#include "MEEBDisplay.hh"

#include <TGraph.h>

ClassImp(MEEBDisplay)

list< TObject* > MEEBDisplay::_list;
map< int, pair<float,float> > MEEBDisplay::_phiLimits;
map< int, pair<float,float> > MEEBDisplay::_etaLimits;
map< int, TPolyLine* > MEEBDisplay::_rzXtals;
int MEEBDisplay::bkgColor=38;
int MEEBDisplay::lineColor=2;
int MEEBDisplay::lineWidth=2;

void
MEEBDisplay::drawEBGlobal()
{
  for( int ii=1; ii<=72; ii++ )
    {      
      TGraph* g_ = MEEBGeom::getGraphBoundary( MEEBGeom::iLMRegion, ii, true );
      g_->SetLineWidth(1);
      g_->Draw("LSame");
    }
  for( int ii=1; ii<=36; ii++ )
    {
      TGraph* g_ = MEEBGeom::getGraphBoundary( MEEBGeom::iSuperModule, ii, true );
      g_->SetLineWidth(2);
      g_->Draw("LSame");
   } 
}

void
MEEBDisplay::drawEBLocal()
{
  TGraph* g_;

  g_ = MEEBGeom::getGraphBoundary( MEEBGeom::iLMRegion, 1, false );
  g_->SetLineWidth(1);
  g_->Draw("LSame");
  
  g_ = MEEBGeom::getGraphBoundary( MEEBGeom::iSuperModule, 1, false );
  g_->SetLineWidth(2);
  g_->Draw("LSame");
}

void 
MEEBDisplay::drawEB()
{
  refresh();
  for( int iSM=1; iSM<=36; iSM++ ) 
    {
      drawSM( iSM );
      for( int iX=0; iX<=16; iX++ )
	{
	  for( int iY=0; iY<=3; iY++ )
	    {
	      drawTT( iSM, iX, iY );
	    }
	}
    }
}

void 
MEEBDisplay::drawSM(  int iSM, float shift )
{
  TPolyLine* pline = getSMPolyLine( iSM, shift );
  registerTObject( pline );
  if( pline==0 ) return;
  
  if( bkgColor>=0 ) pline->SetFillColor( bkgColor );
  pline->SetLineColor( lineColor );
  pline->SetLineWidth( lineWidth );
  if( bkgColor>=0 ) pline->Draw("f"); 
  pline->Draw();
  
  if( shift==0 )
    {
      if( iSM==1 || iSM==19 )  drawSM( iSM, 2. );
    }
}

void 
MEEBDisplay::drawTT(  int iSM, 
		      MEEBGeom::EBTTLocalCoord iX, 
		      MEEBGeom::EBTTLocalCoord iY, float shift )
{
  TPolyLine* pline = getTTPolyLine( iSM, iX, iY, shift );
  registerTObject( pline );
  if( pline==0 ) return;
  
  if( bkgColor>=0 ) pline->SetFillColor( bkgColor );
  pline->SetLineColor( lineColor );
  pline->SetLineWidth( lineWidth );
  if( bkgColor>=0 ) pline->Draw("f"); 
  pline->Draw();

  if( shift==0 )
    {
      if( iSM==1 || iSM==19 )  drawTT( iSM, iX, iY, 2. );
    }
}

void 
MEEBDisplay::drawXtal( MEEBGeom::EBGlobalCoord ieta, 
		       MEEBGeom::EBGlobalCoord iphi, 
		       int color, float shift )
{
  TPolyLine* pline = getXtalPolyLine( ieta, iphi, shift );
  registerTObject( pline );
  if( pline==0 ) return;
  
  pline->SetFillColor( color  );
  pline->SetLineColor( color );
  pline->SetLineWidth( 0 );
  pline->Draw("f"); pline->Draw();

  if( shift==0 )
    {
      int iSM = MEEBGeom::sm( ieta, iphi );
      if( iSM==1 || iSM==19 )  drawXtal( ieta, iphi, color, 2. );
    }
}

void
MEEBDisplay::drawRz()
{

  setRzXtals();

  for( int ii=1; ii<=85; ii++ )
    {
      _rzXtals[ii]->SetLineColor( kGreen+3 );
      _rzXtals[ii]->SetLineWidth( 1 );
      _rzXtals[ii]->SetFillColor( kGreen-9 );
      _rzXtals[ii]->Draw("f");
      _rzXtals[ii]->Draw();

      TPolyLine* pline_;
      for( int jj=1; jj<5; jj++ )
	{
	  pline_ = (TPolyLine*) _rzXtals[ii]->Clone();
	  for( int kk=0; kk<5; kk++ ) 
	    {
	      pline_->GetX()[kk] *= -1;
	    }
	  pline_->Draw("f");
	  pline_->Draw();
	}
      for( int jj=1; jj<5; jj++ )
	{
	  pline_ = (TPolyLine*) _rzXtals[ii]->Clone();
	  for( int kk=0; kk<5; kk++ ) 
	    {
	      pline_->GetX()[kk] *= -1;
	      pline_->GetY()[kk] *= -1;
	    }
	  pline_->Draw("f");
	  pline_->Draw();
	}
      for( int jj=1; jj<5; jj++ )
	{
	  pline_ = (TPolyLine*) _rzXtals[ii]->Clone();
	  for( int kk=0; kk<5; kk++ ) 
	    {
	      pline_->GetY()[kk] *= -1;
	    }
	  pline_->Draw("f");
	  pline_->Draw();
	} 
    } 
}

MEEBGeom::EtaPhiPoint 
MEEBDisplay::getNode( int iSM, 
		      MEEBGeom::EBTTLocalCoord iX, 
		      MEEBGeom::EBTTLocalCoord iY, 
		      int jx, int jy )
{
  assert( iX>=0 && iX<17);
  assert( iY>=0 && iY<4 );
  assert( jx>=0 && jx<=5 );
  assert( jy>=0 && jy<=5 );

  int ix1 = 5*iX+jx-1;
  int iy1 = 5*iY+jy-1;
  int ix2 = 5*iX+jx;
  int iy2 = 5*iY+jy;

  setSM_2_and_20();

  int iSM0=2;
  if( iSM>18 ) iSM0=20;

  int meta = 0;
  if( jx==0 )
    {
      if( iX==0   ) meta = 1;
      if( iX==5   ) meta = 1;
      if( iX==9   ) meta = 1;
      if( iX==13  ) meta = 1;
    }
  if( jx==5 )
    {
      if( iX==4   ) meta = 2;
      if( iX==8   ) meta = 2;
      if( iX==12  ) meta = 2;
      if( iX==16  ) meta = 2;
    }
  
  int mphi = 0;
  if( iY==0  && jy==0 ) mphi = 1;
  if( iY==3  && jy==5 ) mphi = 2;
  
  int ii1 = 10*iSM0+ix1;
  int ii2 = 10*iSM0+ix2;
  float eta_=0;
  if( meta==0 )
    {
      assert( _etaLimits.count(ii1)==1 );
      assert( _etaLimits.count(ii2)==1 );
      eta_ = 0.5*(_etaLimits[ii2].first+_etaLimits[ii1].second);
    }
  else if( meta==1 )
    {
      assert( _etaLimits.count(ii2)==1 );
      eta_ = _etaLimits[ii2].first;
    }
  else if( meta==2 )
    {
      assert( _etaLimits.count(ii1)==1 );
      eta_ = _etaLimits[ii1].second;
    }
  //  if( SM>18 ) eta_*=-1;

  int jj1 = 10*iSM0+iy1;
  int jj2 = 10*iSM0+iy2;
  float phi_=0;
  if( mphi==0 )
    {
      assert( _phiLimits.count(jj1)==1 );
      assert( _phiLimits.count(jj2)==1 );
      phi_ = 0.5*(_phiLimits[jj2].first+_phiLimits[jj1].second);
    }
  else if( mphi==1 )
    {
      assert( _phiLimits.count(jj2)==1 );
      phi_ = _phiLimits[jj2].first;
    }
  else if( mphi==2 )
    {
      assert( _phiLimits.count(jj1)==1 );
      phi_ = _phiLimits[jj1].second;
    }
  phi_ += (iSM-iSM0)*2./18.;

  return MEEBGeom::EtaPhiPoint(eta_,phi_);
}

TPolyLine* 
MEEBDisplay::getXtalPolyLine( MEEBGeom::EBGlobalCoord ieta, 
			      MEEBGeom::EBGlobalCoord iphi, float shift )
{
  int iSM = MEEBGeom::sm( ieta, iphi );
  pair< MEEBGeom::EBLocalCoord, MEEBGeom::EBLocalCoord > p_ = 
    MEEBGeom::localCoord( ieta, iphi );
  int ix = p_.first;
  int iy = p_.second;
  int iX = ix/5;
  int iY = iy/5;
  int jx = ix - 5*iX;
  int jy = iy - 5*iY;

  float eta[5];
  float phi[5];
  int kx[4] = {0,+1,+1, 0};
  int ky[4] = {0, 0,+1,+1};
  for( int ii=0; ii<4; ii++ )
    {
      MEEBGeom::EtaPhiPoint p_ = getNode( iSM, iX, iY, jx+kx[ii], jy+ky[ii] );
      eta[ii] = p_.first;
      phi[ii] = p_.second + shift;
    }
  eta[4] = eta[0];
  phi[4] = phi[0];
  return new TPolyLine( 5, eta, phi );
}

TPolyLine* 
MEEBDisplay::getTTPolyLine( int iSM, 
			    MEEBGeom::EBTTLocalCoord iX, 
			    MEEBGeom::EBTTLocalCoord iY, float shift )
{
  int jx_[4] = {0,5,5,0};
  int jy_[4] = {0,0,5,5};

  float eta[5];
  float phi[5];
  for( int ii=0; ii<4; ii++ )
    {
      MEEBGeom::EtaPhiPoint p_ = getNode( iSM, iX, iY, jx_[ii], jy_[ii] );
      eta[ii] = p_.first;
      phi[ii] = p_.second + shift;
    }
  eta[4] = eta[0];
  phi[4] = phi[0];

  return new TPolyLine( 5, eta, phi );
}

TPolyLine* 
MEEBDisplay::getSMPolyLine( int iSM, float shift )
{
  float eta[5];
  float phi[5];
  {
    
    MEEBGeom::EtaPhiPoint p_ = getNode( iSM, 0, 0, 0, 0 );
    eta[0] = p_.first;
    phi[0] = p_.second + shift;
  }
  {
    MEEBGeom::EtaPhiPoint p_ = getNode( iSM, 16, 0, 5, 0 );
    eta[1] = p_.first;
    phi[1] = p_.second + shift;
  }
  {
    MEEBGeom::EtaPhiPoint p_ = getNode( iSM, 16, 3, 5, 5 );
    eta[2] = p_.first;
    phi[2] = p_.second + shift;
  }
  {
    MEEBGeom::EtaPhiPoint p_ = getNode( iSM, 0, 3, 0, 5 );
    eta[3] = p_.first;
    phi[3] = p_.second + shift;
  }

  eta[4] = eta[0];
  phi[4] = phi[0];

  return new TPolyLine( 5, eta, phi );  
}

void 
MEEBDisplay::setPhiLimits( int iSM,  
			   MEEBGeom::EBLocalCoord iy, 
			   MEEBGeom::EBGlobalCoord iphi, 
			   float phioverpi_0, float phioverpi_1 )
{  
  assert( iSM==2 );
  //  pair<int,int> p1_(iSM,iy);
  pair<float,float> p2_(phioverpi_0,phioverpi_1);
  _phiLimits.insert( pair< int, pair<float,float> >( 10*iSM+iy, p2_ ) );
  pair<float,float> p20_(phioverpi_1,phioverpi_0);
  _phiLimits.insert( pair< int, pair<float,float> >( 10*(iSM+18)+(19-iy), p20_ ) );
}

void 
MEEBDisplay::setEtaLimits( int iSM,  
			   MEEBGeom::EBLocalCoord ix, 
			   MEEBGeom::EBGlobalCoord ieta, 
			   float eta_0, float eta_1 )
{
  assert( iSM==2 );
  //  pair<int,int> p1_(iSM,ix);
  pair<float,float> p2_(eta_0,eta_1);
  _etaLimits.insert( pair< int, pair<float,float> >( 10*iSM+ix, p2_ ) );
  pair<float,float> p20_(-eta_0,-eta_1);
  _etaLimits.insert( pair< int, pair<float,float> >( 10*(iSM+18)+ix, p20_ ) );
}

void 
MEEBDisplay::setSM_2_and_20()
{
  static bool done=false;
  if( done ) return;
  done = true;

  setPhiLimits(	2,	0,	40,	0.1661,	0.1607);
  setPhiLimits(	2,	1,	39,	0.1606,	0.1552);
  setPhiLimits(	2,	2,	38,	0.1551,	0.1497);
  setPhiLimits(	2,	3,	37,	0.1496,	0.1442);
  setPhiLimits(	2,	4,	36,	0.1441,	0.1387);
  setPhiLimits(	2,	5,	35,	0.1386,	0.1333);
  setPhiLimits(	2,	6,	34,	0.1331,	0.1277);
  setPhiLimits(	2,	7,	33,	0.1276,	0.1223);
  setPhiLimits(	2,	8,	32,	0.1221,	0.1168);
  setPhiLimits(	2,	9,	31,	0.1166,	0.1113);
  setPhiLimits(	2,	10,	30,	0.1111,	0.1058);
  setPhiLimits(	2,	11,	29,	0.1057,	0.1003);
  setPhiLimits(	2,	12,	28,	0.1001,	0.0948);
  setPhiLimits(	2,	13,	27,	0.0947,	0.0893);
  setPhiLimits(	2,	14,	26,	0.0892,	0.0838);
  setPhiLimits(	2,	15,	25,	0.0837,	0.0783);
  setPhiLimits(	2,	16,	24,	0.0782,	0.0728);
  setPhiLimits(	2,	17,	23,	0.0727,	0.0673);
  setPhiLimits(	2,	18,	22,	0.0672,	0.0618);
  setPhiLimits(	2,	19,	21,	0.0617,	0.0564);

  setEtaLimits(	2,	0,	1,	0.0026,	0.0208);
  setEtaLimits(	2,	1,	2,	0.0212,	0.0394);
  setEtaLimits(	2,	2,	3,	0.0397,	0.0579);
  setEtaLimits(	2,	3,	4,	0.0582,	0.0764);
  setEtaLimits(	2,	4,	5,	0.0767,	0.0949);
  setEtaLimits(	2,	5,	6,	0.0952,	0.1123);
  setEtaLimits(	2,	6,	7,	0.1126,	0.1297);
  setEtaLimits(	2,	7,	8,	0.1299,	0.1471);
  setEtaLimits(	2,	8,	9,	0.1473,	0.1644);
  setEtaLimits(	2,	9,	10,	0.1646,	0.1817);
  setEtaLimits(	2,	10,	11,	0.1819,	0.1991);
  setEtaLimits(	2,	11,	12,	0.1993,	0.2165);
  setEtaLimits(	2,	12,	13,	0.2166,	0.2338);
  setEtaLimits(	2,	13,	14,	0.2339,	0.2511);
  setEtaLimits(	2,	14,	15,	0.2513,	0.2685);
  setEtaLimits(	2,	15,	16,	0.2686,	0.2859);
  setEtaLimits(	2,	16,	17,	0.2860,	0.3032);
  setEtaLimits(	2,	17,	18,	0.3033,	0.3206);
  setEtaLimits(	2,	18,	19,	0.3207,	0.3379);
  setEtaLimits(	2,	19,	20,	0.3380,	0.3553);
  setEtaLimits(	2,	20,	21,	0.3553,	0.3727);
  setEtaLimits(	2,	21,	22,	0.3727,	0.3901);
  setEtaLimits(	2,	22,	23,	0.3901,	0.4075);
  setEtaLimits(	2,	23,	24,	0.4074,	0.4248);
  setEtaLimits(	2,	24,	25,	0.4248,	0.4422);
  setEtaLimits(	2,	25,	26,	0.4468,	0.4642);
  setEtaLimits(	2,	26,	27,	0.4641,	0.4814);
  setEtaLimits(	2,	27,	28,	0.4813,	0.4987);
  setEtaLimits(	2,	28,	29,	0.4986,	0.5159);
  setEtaLimits(	2,	29,	30,	0.5158,	0.5331);
  setEtaLimits(	2,	30,	31,	0.5330,	0.5503);
  setEtaLimits(	2,	31,	32,	0.5501,	0.5674);
  setEtaLimits(	2,	32,	33,	0.5673,	0.5846);
  setEtaLimits(	2,	33,	34,	0.5844,	0.6017);
  setEtaLimits(	2,	34,	35,	0.6014,	0.6188);
  setEtaLimits(	2,	35,	36,	0.6185,	0.6359);
  setEtaLimits(	2,	36,	37,	0.6357,	0.6531);
  setEtaLimits(	2,	37,	38,	0.6528,	0.6702);
  setEtaLimits(	2,	38,	39,	0.6699,	0.6873);
  setEtaLimits(	2,	39,	40,	0.6870,	0.7044);
  setEtaLimits(	2,	40,	41,	0.7041,	0.7216);
  setEtaLimits(	2,	41,	42,	0.7212,	0.7387);
  setEtaLimits(	2,	42,	43,	0.7384,	0.7559);
  setEtaLimits(	2,	43,	44,	0.7555,	0.7730);
  setEtaLimits(	2,	44,	45,	0.7726,	0.7901);
  setEtaLimits(	2,	45,	46,	0.7948,	0.8125);
  setEtaLimits(	2,	46,	47,	0.8120,	0.8297);
  setEtaLimits(	2,	47,	48,	0.8292,	0.8468);
  setEtaLimits(	2,	48,	49,	0.8463,	0.8640);
  setEtaLimits(	2,	49,	50,	0.8635,	0.8811);
  setEtaLimits(	2,	50,	51,	0.8806,	0.8983);
  setEtaLimits(	2,	51,	52,	0.8978,	0.9155);
  setEtaLimits(	2,	52,	53,	0.9149,	0.9326);
  setEtaLimits(	2,	53,	54,	0.9320,	0.9497);
  setEtaLimits(	2,	54,	55,	0.9491,	0.9668);
  setEtaLimits(	2,	55,	56,	0.9662,	0.9840);
  setEtaLimits(	2,	56,	57,	0.9833,	1.0011);
  setEtaLimits(	2,	57,	58,	1.0004,	1.0183);
  setEtaLimits(	2,	58,	59,	1.0175,	1.0354);
  setEtaLimits(	2,	59,	60,	1.0346,	1.0525);
  setEtaLimits(	2,	60,	61,	1.0517,	1.0697);
  setEtaLimits(	2,	61,	62,	1.0689,	1.0869);
  setEtaLimits(	2,	62,	63,	1.0860,	1.1040);
  setEtaLimits(	2,	63,	64,	1.1032,	1.1211);
  setEtaLimits(	2,	64,	65,	1.1203,	1.1382);
  setEtaLimits(	2,	65,	66,	1.1426,	1.1607);
  setEtaLimits(	2,	66,	67,	1.1597,	1.1779);
  setEtaLimits(	2,	67,	68,	1.1769,	1.1950);
  setEtaLimits(	2,	68,	69,	1.1940,	1.2121);
  setEtaLimits(	2,	69,	70,	1.2111,	1.2292);
  setEtaLimits(	2,	70,	71,	1.2281,	1.2464);
  setEtaLimits(	2,	71,	72,	1.2453,	1.2636);
  setEtaLimits(	2,	72,	73,	1.2624,	1.2807);
  setEtaLimits(	2,	73,	74,	1.2795,	1.2978);
  setEtaLimits(	2,	74,	75,	1.2966,	1.3149);
  setEtaLimits(	2,	75,	76,	1.3137,	1.3321);
  setEtaLimits(	2,	76,	77,	1.3309,	1.3493);
  setEtaLimits(	2,	77,	78,	1.3480,	1.3664);
  setEtaLimits(	2,	78,	79,	1.3651,	1.3835);
  setEtaLimits(	2,	79,	80,	1.3822,	1.4006);
  setEtaLimits(	2,	80,	81,	1.3993,	1.4178);
  setEtaLimits(	2,	81,	82,	1.4165,	1.4350);
  setEtaLimits(	2,	82,	83,	1.4336,	1.4522);
  setEtaLimits(	2,	83,	84,	1.4507,	1.4693);
  setEtaLimits(	2,	84,	85,	1.4678,	1.4864);
}

void
MEEBDisplay::setRzXtals()
{
  static bool done=false;
  if( done ) return;
  done = true;

  cout << "SetRzXtals " << endl;

  // eta=1
  double ebxx_1[5]; double ebyy_1[5];
  ebxx_1[0]= 2.693; ebyy_1[0]= 129.124;
  ebxx_1[1]= 0.338; ebyy_1[1]= 129.124;
  ebxx_1[2]= 0.338; ebyy_1[2]= 152.100;
  ebxx_1[3]= 2.883; ebyy_1[3]= 152.100;
  ebxx_1[4]=   ebxx_1[0]; ebyy_1[4]=   ebyy_1[0];;
  _rzXtals[1] = new TPolyLine( 5, ebxx_1,ebyy_1);

  // eta=2
  double ebxx_2[5]; double ebyy_2[5];
  ebxx_2[0]= 5.089; ebyy_2[0]= 129.124;
  ebxx_2[1]= 2.735; ebyy_2[1]= 129.144;
  ebxx_2[2]= 2.924; ebyy_2[2]= 152.119;
  ebxx_2[3]= 5.469; ebyy_2[3]= 152.098;
  ebxx_2[4]=   ebxx_2[0]; ebyy_2[4]=   ebyy_2[0];;
  _rzXtals[2] = new TPolyLine( 5, ebxx_2,ebyy_2);

  // eta=3
  double ebxx_3[5]; double ebyy_3[5];
  ebxx_3[0]= 7.486; ebyy_3[0]= 129.124;
  ebxx_3[1]= 5.131; ebyy_3[1]= 129.163;
  ebxx_3[2]= 5.510; ebyy_3[2]= 152.136;
  ebxx_3[3]= 8.054; ebyy_3[3]= 152.094;
  ebxx_3[4]=   ebxx_3[0]; ebyy_3[4]=   ebyy_3[0];;
  _rzXtals[3] = new TPolyLine( 5, ebxx_3,ebyy_3);

  // eta=4
  double ebxx_4[5]; double ebyy_4[5];
  ebxx_4[0]= 9.883; ebyy_4[0]= 129.124;
  ebxx_4[1]= 7.529; ebyy_4[1]= 129.182;
  ebxx_4[2]= 8.096; ebyy_4[2]= 152.151;
  ebxx_4[3]= 10.640; ebyy_4[3]= 152.089;
  ebxx_4[4]=   ebxx_4[0]; ebyy_4[4]=   ebyy_4[0];;
  _rzXtals[4] = new TPolyLine( 5, ebxx_4,ebyy_4);

  // eta=5
  double ebxx_5[5]; double ebyy_5[5];
  ebxx_5[0]= 12.280; ebyy_5[0]= 129.124;
  ebxx_5[1]= 9.927; ebyy_5[1]= 129.202;
  ebxx_5[2]= 10.683; ebyy_5[2]= 152.165;
  ebxx_5[3]= 13.227; ebyy_5[3]= 152.082;
  ebxx_5[4]=   ebxx_5[0]; ebyy_5[4]=   ebyy_5[0];;
  _rzXtals[5] = new TPolyLine( 5, ebxx_5,ebyy_5);

  // eta=6
  double ebxx_6[5]; double ebyy_6[5];
  ebxx_6[0]= 14.541; ebyy_6[0]= 129.124;
  ebxx_6[1]= 12.325; ebyy_6[1]= 129.215;
  ebxx_6[2]= 13.271; ebyy_6[2]= 152.172;
  ebxx_6[3]= 15.888; ebyy_6[3]= 152.064;
  ebxx_6[4]=   ebxx_6[0]; ebyy_6[4]=   ebyy_6[0];;
  _rzXtals[6] = new TPolyLine( 5, ebxx_6,ebyy_6);

  // eta=7
  double ebxx_7[5]; double ebyy_7[5];
  ebxx_7[0]= 16.801; ebyy_7[0]= 129.124;
  ebxx_7[1]= 14.587; ebyy_7[1]= 129.254;
  ebxx_7[2]= 15.933; ebyy_7[2]= 152.191;
  ebxx_7[3]= 18.547; ebyy_7[3]= 152.038;
  ebxx_7[4]=   ebxx_7[0]; ebyy_7[4]=   ebyy_7[0];;
  _rzXtals[7] = new TPolyLine( 5, ebxx_7,ebyy_7);

  // eta=8
  double ebxx_8[5]; double ebyy_8[5];
  ebxx_8[0]= 19.063; ebyy_8[0]= 129.124;
  ebxx_8[1]= 16.852; ebyy_8[1]= 129.292;
  ebxx_8[2]= 18.597; ebyy_8[2]= 152.202;
  ebxx_8[3]= 21.208; ebyy_8[3]= 152.004;
  ebxx_8[4]=   ebxx_8[0]; ebyy_8[4]=   ebyy_8[0];;
  _rzXtals[8] = new TPolyLine( 5, ebxx_8,ebyy_8);

  // eta=9
  double ebxx_9[5]; double ebyy_9[5];
  ebxx_9[0]= 21.329; ebyy_9[0]= 129.124;
  ebxx_9[1]= 19.121; ebyy_9[1]= 129.331;
  ebxx_9[2]= 21.265; ebyy_9[2]= 152.207;
  ebxx_9[3]= 23.872; ebyy_9[3]= 151.963;
  ebxx_9[4]=   ebxx_9[0]; ebyy_9[4]=   ebyy_9[0];;
  _rzXtals[9] = new TPolyLine( 5, ebxx_9,ebyy_9);

  // eta=10
  double ebxx_10[5]; double ebyy_10[5];
  ebxx_10[0]= 23.599; ebyy_10[0]= 129.124;
  ebxx_10[1]= 21.395; ebyy_10[1]= 129.369;
  ebxx_10[2]= 23.937; ebyy_10[2]= 152.204;
  ebxx_10[3]= 26.540; ebyy_10[3]= 151.915;
  ebxx_10[4]=   ebxx_10[0]; ebyy_10[4]=   ebyy_10[0];;
  _rzXtals[10] = new TPolyLine( 5, ebxx_10,ebyy_10);

  // eta=11
  double ebxx_11[5]; double ebyy_11[5];
  ebxx_11[0]= 25.887; ebyy_11[0]= 129.124;
  ebxx_11[1]= 23.676; ebyy_11[1]= 129.409;
  ebxx_11[2]= 26.615; ebyy_11[2]= 152.197;
  ebxx_11[3]= 29.218; ebyy_11[3]= 151.862;
  ebxx_11[4]=   ebxx_11[0]; ebyy_11[4]=   ebyy_11[0];;
  _rzXtals[11] = new TPolyLine( 5, ebxx_11,ebyy_11);

  // eta=12
  double ebxx_12[5]; double ebyy_12[5];
  ebxx_12[0]= 28.180; ebyy_12[0]= 129.124;
  ebxx_12[1]= 25.973; ebyy_12[1]= 129.447;
  ebxx_12[2]= 29.303; ebyy_12[2]= 152.181;
  ebxx_12[3]= 31.900; ebyy_12[3]= 151.801;
  ebxx_12[4]=   ebxx_12[0]; ebyy_12[4]=   ebyy_12[0];;
  _rzXtals[12] = new TPolyLine( 5, ebxx_12,ebyy_12);

  // eta=13
  double ebxx_13[5]; double ebyy_13[5];
  ebxx_13[0]= 30.478; ebyy_13[0]= 129.124;
  ebxx_13[1]= 28.278; ebyy_13[1]= 129.484;
  ebxx_13[2]= 31.997; ebyy_13[2]= 152.158;
  ebxx_13[3]= 34.587; ebyy_13[3]= 151.734;
  ebxx_13[4]=   ebxx_13[0]; ebyy_13[4]=   ebyy_13[0];;
  _rzXtals[13] = new TPolyLine( 5, ebxx_13,ebyy_13);

  // eta=14
  double ebxx_14[5]; double ebyy_14[5];
  ebxx_14[0]= 32.783; ebyy_14[0]= 129.124;
  ebxx_14[1]= 30.589; ebyy_14[1]= 129.522;
  ebxx_14[2]= 34.697; ebyy_14[2]= 152.129;
  ebxx_14[3]= 37.279; ebyy_14[3]= 151.660;
  ebxx_14[4]=   ebxx_14[0]; ebyy_14[4]=   ebyy_14[0];;
  _rzXtals[14] = new TPolyLine( 5, ebxx_14,ebyy_14);

  // eta=15
  double ebxx_15[5]; double ebyy_15[5];
  ebxx_15[0]= 35.096; ebyy_15[0]= 129.124;
  ebxx_15[1]= 32.909; ebyy_15[1]= 129.559;
  ebxx_15[2]= 37.404; ebyy_15[2]= 152.093;
  ebxx_15[3]= 39.978; ebyy_15[3]= 151.580;
  ebxx_15[4]=   ebxx_15[0]; ebyy_15[4]=   ebyy_15[0];;
  _rzXtals[15] = new TPolyLine( 5, ebxx_15,ebyy_15);

  // eta=16
  double ebxx_16[5]; double ebyy_16[5];
  ebxx_16[0]= 37.431; ebyy_16[0]= 129.124;
  ebxx_16[1]= 35.240; ebyy_16[1]= 129.599;
  ebxx_16[2]= 40.120; ebyy_16[2]= 152.052;
  ebxx_16[3]= 42.688; ebyy_16[3]= 151.495;
  ebxx_16[4]=   ebxx_16[0]; ebyy_16[4]=   ebyy_16[0];;
  _rzXtals[16] = new TPolyLine( 5, ebxx_16,ebyy_16);

  // eta=17
  double ebxx_17[5]; double ebyy_17[5];
  ebxx_17[0]= 39.773; ebyy_17[0]= 129.124;
  ebxx_17[1]= 37.591; ebyy_17[1]= 129.636;
  ebxx_17[2]= 42.846; ebyy_17[2]= 152.004;
  ebxx_17[3]= 45.405; ebyy_17[3]= 151.404;
  ebxx_17[4]=   ebxx_17[0]; ebyy_17[4]=   ebyy_17[0];;
  _rzXtals[17] = new TPolyLine( 5, ebxx_17,ebyy_17);

  // eta=18
  double ebxx_18[5]; double ebyy_18[5];
  ebxx_18[0]= 42.125; ebyy_18[0]= 129.124;
  ebxx_18[1]= 39.951; ebyy_18[1]= 129.672;
  ebxx_18[2]= 45.581; ebyy_18[2]= 151.950;
  ebxx_18[3]= 48.129; ebyy_18[3]= 151.307;
  ebxx_18[4]=   ebxx_18[0]; ebyy_18[4]=   ebyy_18[0];;
  _rzXtals[18] = new TPolyLine( 5, ebxx_18,ebyy_18);

  // eta=19
  double ebxx_19[5]; double ebyy_19[5];
  ebxx_19[0]= 44.487; ebyy_19[0]= 129.124;
  ebxx_19[1]= 42.323; ebyy_19[1]= 129.709;
  ebxx_19[2]= 48.325; ebyy_19[2]= 151.889;
  ebxx_19[3]= 50.862; ebyy_19[3]= 151.203;
  ebxx_19[4]=   ebxx_19[0]; ebyy_19[4]=   ebyy_19[0];;
  _rzXtals[19] = new TPolyLine( 5, ebxx_19,ebyy_19);

  // eta=20
  double ebxx_20[5]; double ebyy_20[5];
  ebxx_20[0]= 46.861; ebyy_20[0]= 129.124;
  ebxx_20[1]= 44.706; ebyy_20[1]= 129.745;
  ebxx_20[2]= 51.080; ebyy_20[2]= 151.821;
  ebxx_20[3]= 53.604; ebyy_20[3]= 151.094;
  ebxx_20[4]=   ebxx_20[0]; ebyy_20[4]=   ebyy_20[0];;
  _rzXtals[20] = new TPolyLine( 5, ebxx_20,ebyy_20);

  // eta=21
  double ebxx_21[5]; double ebyy_21[5];
  ebxx_21[0]= 49.262; ebyy_21[0]= 129.124;
  ebxx_21[1]= 47.105; ebyy_21[1]= 129.785;
  ebxx_21[2]= 53.847; ebyy_21[2]= 151.752;
  ebxx_21[3]= 56.362; ebyy_21[3]= 150.981;
  ebxx_21[4]=   ebxx_21[0]; ebyy_21[4]=   ebyy_21[0];;
  _rzXtals[21] = new TPolyLine( 5, ebxx_21,ebyy_21);

  // eta=22
  double ebxx_22[5]; double ebyy_22[5];
  ebxx_22[0]= 51.674; ebyy_22[0]= 129.124;
  ebxx_22[1]= 49.528; ebyy_22[1]= 129.820;
  ebxx_22[2]= 56.626; ebyy_22[2]= 151.674;
  ebxx_22[3]= 59.128; ebyy_22[3]= 150.863;
  ebxx_22[4]=   ebxx_22[0]; ebyy_22[4]=   ebyy_22[0];;
  _rzXtals[22] = new TPolyLine( 5, ebxx_22,ebyy_22);

  // eta=23
  double ebxx_23[5]; double ebyy_23[5];
  ebxx_23[0]= 54.099; ebyy_23[0]= 129.124;
  ebxx_23[1]= 51.965; ebyy_23[1]= 129.854;
  ebxx_23[2]= 59.418; ebyy_23[2]= 151.591;
  ebxx_23[3]= 61.906; ebyy_23[3]= 150.739;
  ebxx_23[4]=   ebxx_23[0]; ebyy_23[4]=   ebyy_23[0];;
  _rzXtals[23] = new TPolyLine( 5, ebxx_23,ebyy_23);

  // eta=24
  double ebxx_24[5]; double ebyy_24[5];
  ebxx_24[0]= 56.539; ebyy_24[0]= 129.124;
  ebxx_24[1]= 54.416; ebyy_24[1]= 129.889;
  ebxx_24[2]= 62.221; ebyy_24[2]= 151.502;
  ebxx_24[3]= 64.696; ebyy_24[3]= 150.610;
  ebxx_24[4]=   ebxx_24[0]; ebyy_24[4]=   ebyy_24[0];;
  _rzXtals[24] = new TPolyLine( 5, ebxx_24,ebyy_24);

  // eta=25
  double ebxx_25[5]; double ebyy_25[5];
  ebxx_25[0]= 58.993; ebyy_25[0]= 129.124;
  ebxx_25[1]= 56.883; ebyy_25[1]= 129.923;
  ebxx_25[2]= 65.038; ebyy_25[2]= 151.406;
  ebxx_25[3]= 67.498; ebyy_25[3]= 150.475;
  ebxx_25[4]=   ebxx_25[0]; ebyy_25[4]=   ebyy_25[0];;
  _rzXtals[25] = new TPolyLine( 5, ebxx_25,ebyy_25);

  // eta=26
  double ebxx_26[5]; double ebyy_26[5];
  ebxx_26[0]= 62.135; ebyy_26[0]= 129.124;
  ebxx_26[1]= 60.039; ebyy_26[1]= 129.957;
  ebxx_26[2]= 68.543; ebyy_26[2]= 151.305;
  ebxx_26[3]= 70.972; ebyy_26[3]= 150.340;
  ebxx_26[4]=   ebxx_26[0]; ebyy_26[4]=   ebyy_26[0];;
  _rzXtals[26] = new TPolyLine( 5, ebxx_26,ebyy_26);

  // eta=27
  double ebxx_27[5]; double ebyy_27[5];
  ebxx_27[0]= 64.620; ebyy_27[0]= 129.124;
  ebxx_27[1]= 62.537; ebyy_27[1]= 129.990;
  ebxx_27[2]= 71.372; ebyy_27[2]= 151.203;
  ebxx_27[3]= 73.786; ebyy_27[3]= 150.200;
  ebxx_27[4]=   ebxx_27[0]; ebyy_27[4]=   ebyy_27[0];;
  _rzXtals[27] = new TPolyLine( 5, ebxx_27,ebyy_27);

  // eta=28
  double ebxx_28[5]; double ebyy_28[5];
  ebxx_28[0]= 67.121; ebyy_28[0]= 129.124;
  ebxx_28[1]= 65.052; ebyy_28[1]= 130.022;
  ebxx_28[2]= 74.217; ebyy_28[2]= 151.095;
  ebxx_28[3]= 76.615; ebyy_28[3]= 150.054;
  ebxx_28[4]=   ebxx_28[0]; ebyy_28[4]=   ebyy_28[0];;
  _rzXtals[28] = new TPolyLine( 5, ebxx_28,ebyy_28);

  // eta=29
  double ebxx_29[5]; double ebyy_29[5];
  ebxx_29[0]= 69.639; ebyy_29[0]= 129.124;
  ebxx_29[1]= 67.585; ebyy_29[1]= 130.054;
  ebxx_29[2]= 77.077; ebyy_29[2]= 150.982;
  ebxx_29[3]= 79.458; ebyy_29[3]= 149.904;
  ebxx_29[4]=   ebxx_29[0]; ebyy_29[4]=   ebyy_29[0];;
  _rzXtals[29] = new TPolyLine( 5, ebxx_29,ebyy_29);

  // eta=30
  double ebxx_30[5]; double ebyy_30[5];
  ebxx_30[0]= 72.176; ebyy_30[0]= 129.124;
  ebxx_30[1]= 70.136; ebyy_30[1]= 130.086;
  ebxx_30[2]= 79.953; ebyy_30[2]= 150.864;
  ebxx_30[3]= 82.318; ebyy_30[3]= 149.749;
  ebxx_30[4]=   ebxx_30[0]; ebyy_30[4]=   ebyy_30[0];;
  _rzXtals[30] = new TPolyLine( 5, ebxx_30,ebyy_30);

  // eta=31
  double ebxx_31[5]; double ebyy_31[5];
  ebxx_31[0]= 74.727; ebyy_31[0]= 129.124;
  ebxx_31[1]= 72.707; ebyy_31[1]= 130.115;
  ebxx_31[2]= 82.847; ebyy_31[2]= 150.738;
  ebxx_31[3]= 85.173; ebyy_31[3]= 149.596;
  ebxx_31[4]=   ebxx_31[0]; ebyy_31[4]=   ebyy_31[0];;
  _rzXtals[31] = new TPolyLine( 5, ebxx_31,ebyy_31);

  // eta=32
  double ebxx_32[5]; double ebyy_32[5];
  ebxx_32[0]= 77.296; ebyy_32[0]= 129.124;
  ebxx_32[1]= 75.291; ebyy_32[1]= 130.145;
  ebxx_32[2]= 85.735; ebyy_32[2]= 150.615;
  ebxx_32[3]= 88.044; ebyy_32[3]= 149.439;
  ebxx_32[4]=   ebxx_32[0]; ebyy_32[4]=   ebyy_32[0];;
  _rzXtals[32] = new TPolyLine( 5, ebxx_32,ebyy_32);

  // eta=33
  double ebxx_33[5]; double ebyy_33[5];
  ebxx_33[0]= 79.884; ebyy_33[0]= 129.124;
  ebxx_33[1]= 77.895; ebyy_33[1]= 130.174;
  ebxx_33[2]= 88.641; ebyy_33[2]= 150.488;
  ebxx_33[3]= 90.933; ebyy_33[3]= 149.278;
  ebxx_33[4]=   ebxx_33[0]; ebyy_33[4]=   ebyy_33[0];;
  _rzXtals[33] = new TPolyLine( 5, ebxx_33,ebyy_33);

  // eta=34
  double ebxx_34[5]; double ebyy_34[5];
  ebxx_34[0]= 82.493; ebyy_34[0]= 129.124;
  ebxx_34[1]= 80.520; ebyy_34[1]= 130.204;
  ebxx_34[2]= 91.566; ebyy_34[2]= 150.356;
  ebxx_34[3]= 93.840; ebyy_34[3]= 149.112;
  ebxx_34[4]=   ebxx_34[0]; ebyy_34[4]=   ebyy_34[0];;
  _rzXtals[34] = new TPolyLine( 5, ebxx_34,ebyy_34);

  // eta=35
  double ebxx_35[5]; double ebyy_35[5];
  ebxx_35[0]= 85.124; ebyy_35[0]= 129.124;
  ebxx_35[1]= 83.167; ebyy_35[1]= 130.233;
  ebxx_35[2]= 94.511; ebyy_35[2]= 150.219;
  ebxx_35[3]= 96.766; ebyy_35[3]= 148.942;
  ebxx_35[4]=   ebxx_35[0]; ebyy_35[4]=   ebyy_35[0];;
  _rzXtals[35] = new TPolyLine( 5, ebxx_35,ebyy_35);

  // eta=36
  double ebxx_36[5]; double ebyy_36[5];
  ebxx_36[0]= 87.793; ebyy_36[0]= 129.124;
  ebxx_36[1]= 85.841; ebyy_36[1]= 130.268;
  ebxx_36[2]= 97.481; ebyy_36[2]= 150.085;
  ebxx_36[3]= 99.714; ebyy_36[3]= 148.776;
  ebxx_36[4]=   ebxx_36[0]; ebyy_36[4]=   ebyy_36[0];;
  _rzXtals[36] = new TPolyLine( 5, ebxx_36,ebyy_36);

  // eta=37
  double ebxx_37[5]; double ebyy_37[5];
  ebxx_37[0]= 90.484; ebyy_37[0]= 129.124;
  ebxx_37[1]= 88.548; ebyy_37[1]= 130.295;
  ebxx_37[2]= 100.467; ebyy_37[2]= 149.946;
  ebxx_37[3]= 102.681; ebyy_37[3]= 148.606;
  ebxx_37[4]=   ebxx_37[0]; ebyy_37[4]=   ebyy_37[0];;
  _rzXtals[37] = new TPolyLine( 5, ebxx_37,ebyy_37);

  // eta=38
  double ebxx_38[5]; double ebyy_38[5];
  ebxx_38[0]= 93.198; ebyy_38[0]= 129.124;
  ebxx_38[1]= 91.279; ebyy_38[1]= 130.322;
  ebxx_38[2]= 103.475; ebyy_38[2]= 149.803;
  ebxx_38[3]= 105.669; ebyy_38[3]= 148.432;
  ebxx_38[4]=   ebxx_38[0]; ebyy_38[4]=   ebyy_38[0];;
  _rzXtals[38] = new TPolyLine( 5, ebxx_38,ebyy_38);

  // eta=39
  double ebxx_39[5]; double ebyy_39[5];
  ebxx_39[0]= 95.936; ebyy_39[0]= 129.124;
  ebxx_39[1]= 94.035; ebyy_39[1]= 130.349;
  ebxx_39[2]= 106.504; ebyy_39[2]= 149.656;
  ebxx_39[3]= 108.679; ebyy_39[3]= 148.254;
  ebxx_39[4]=   ebxx_39[0]; ebyy_39[4]=   ebyy_39[0];;
  _rzXtals[39] = new TPolyLine( 5, ebxx_39,ebyy_39);

  // eta=40
  double ebxx_40[5]; double ebyy_40[5];
  ebxx_40[0]= 98.700; ebyy_40[0]= 129.124;
  ebxx_40[1]= 96.816; ebyy_40[1]= 130.376;
  ebxx_40[2]= 109.557; ebyy_40[2]= 149.505;
  ebxx_40[3]= 111.712; ebyy_40[3]= 148.072;
  ebxx_40[4]=   ebxx_40[0]; ebyy_40[4]=   ebyy_40[0];;
  _rzXtals[40] = new TPolyLine( 5, ebxx_40,ebyy_40);

  // eta=41
  double ebxx_41[5]; double ebyy_41[5];
  ebxx_41[0]= 101.510; ebyy_41[0]= 129.124;
  ebxx_41[1]= 99.632; ebyy_41[1]= 130.411;
  ebxx_41[2]= 112.642; ebyy_41[2]= 149.358;
  ebxx_41[3]= 114.775; ebyy_41[3]= 147.896;
  ebxx_41[4]=   ebxx_41[0]; ebyy_41[4]=   ebyy_41[0];;
  _rzXtals[41] = new TPolyLine( 5, ebxx_41,ebyy_41);

  // eta=42
  double ebxx_42[5]; double ebyy_42[5];
  ebxx_42[0]= 104.345; ebyy_42[0]= 129.124;
  ebxx_42[1]= 102.484; ebyy_42[1]= 130.436;
  ebxx_42[2]= 115.747; ebyy_42[2]= 149.207;
  ebxx_42[3]= 117.860; ebyy_42[3]= 147.717;
  ebxx_42[4]=   ebxx_42[0]; ebyy_42[4]=   ebyy_42[0];;
  _rzXtals[42] = new TPolyLine( 5, ebxx_42,ebyy_42);

  // eta=43
  double ebxx_43[5]; double ebyy_43[5];
  ebxx_43[0]= 107.207; ebyy_43[0]= 129.124;
  ebxx_43[1]= 105.364; ebyy_43[1]= 130.461;
  ebxx_43[2]= 118.877; ebyy_43[2]= 149.053;
  ebxx_43[3]= 120.971; ebyy_43[3]= 147.534;
  ebxx_43[4]=   ebxx_43[0]; ebyy_43[4]=   ebyy_43[0];;
  _rzXtals[43] = new TPolyLine( 5, ebxx_43,ebyy_43);

  // eta=44
  double ebxx_44[5]; double ebyy_44[5];
  ebxx_44[0]= 110.098; ebyy_44[0]= 129.124;
  ebxx_44[1]= 108.273; ebyy_44[1]= 130.485;
  ebxx_44[2]= 122.034; ebyy_44[2]= 148.895;
  ebxx_44[3]= 124.107; ebyy_44[3]= 147.349;
  ebxx_44[4]=   ebxx_44[0]; ebyy_44[4]=   ebyy_44[0];;
  _rzXtals[44] = new TPolyLine( 5, ebxx_44,ebyy_44);

  // eta=45
  double ebxx_45[5]; double ebyy_45[5];
  ebxx_45[0]= 113.017; ebyy_45[0]= 129.124;
  ebxx_45[1]= 111.211; ebyy_45[1]= 130.510;
  ebxx_45[2]= 125.218; ebyy_45[2]= 148.733;
  ebxx_45[3]= 127.270; ebyy_45[3]= 147.159;
  ebxx_45[4]=   ebxx_45[0]; ebyy_45[4]=   ebyy_45[0];;
  _rzXtals[45] = new TPolyLine( 5, ebxx_45,ebyy_45);

  // eta=46
  double ebxx_46[5]; double ebyy_46[5];
  ebxx_46[0]= 116.882; ebyy_46[0]= 129.124;
  ebxx_46[1]= 115.074; ebyy_46[1]= 130.550;
  ebxx_46[2]= 129.324; ebyy_46[2]= 148.584;
  ebxx_46[3]= 131.362; ebyy_46[3]= 146.977;
  ebxx_46[4]=   ebxx_46[0]; ebyy_46[4]=   ebyy_46[0];;
  _rzXtals[46] = new TPolyLine( 5, ebxx_46,ebyy_46);

  // eta=47
  double ebxx_47[5]; double ebyy_47[5];
  ebxx_47[0]= 119.895; ebyy_47[0]= 129.124;
  ebxx_47[1]= 118.106; ebyy_47[1]= 130.572;
  ebxx_47[2]= 132.584; ebyy_47[2]= 148.425;
  ebxx_47[3]= 134.602; ebyy_47[3]= 146.792;
  ebxx_47[4]=   ebxx_47[0]; ebyy_47[4]=   ebyy_47[0];;
  _rzXtals[47] = new TPolyLine( 5, ebxx_47,ebyy_47);

  // eta=48
  double ebxx_48[5]; double ebyy_48[5];
  ebxx_48[0]= 122.941; ebyy_48[0]= 129.124;
  ebxx_48[1]= 121.169; ebyy_48[1]= 130.595;
  ebxx_48[2]= 135.874; ebyy_48[2]= 148.262;
  ebxx_48[3]= 137.870; ebyy_48[3]= 146.603;
  ebxx_48[4]=   ebxx_48[0]; ebyy_48[4]=   ebyy_48[0];;
  _rzXtals[48] = new TPolyLine( 5, ebxx_48,ebyy_48);

  // eta=49
  double ebxx_49[5]; double ebyy_49[5];
  ebxx_49[0]= 126.018; ebyy_49[0]= 129.124;
  ebxx_49[1]= 124.266; ebyy_49[1]= 130.617;
  ebxx_49[2]= 139.194; ebyy_49[2]= 148.096;
  ebxx_49[3]= 141.169; ebyy_49[3]= 146.412;
  ebxx_49[4]=   ebxx_49[0]; ebyy_49[4]=   ebyy_49[0];;
  _rzXtals[49] = new TPolyLine( 5, ebxx_49,ebyy_49);

  // eta=50
  double ebxx_50[5]; double ebyy_50[5];
  ebxx_50[0]= 129.130; ebyy_50[0]= 129.124;
  ebxx_50[1]= 127.397; ebyy_50[1]= 130.639;
  ebxx_50[2]= 142.546; ebyy_50[2]= 147.927;
  ebxx_50[3]= 144.500; ebyy_50[3]= 146.219;
  ebxx_50[4]=   ebxx_50[0]; ebyy_50[4]=   ebyy_50[0];;
  _rzXtals[50] = new TPolyLine( 5, ebxx_50,ebyy_50);

  // eta=51
  double ebxx_51[5]; double ebyy_51[5];
  ebxx_51[0]= 132.289; ebyy_51[0]= 129.124;
  ebxx_51[1]= 130.571; ebyy_51[1]= 130.666;
  ebxx_51[2]= 145.939; ebyy_51[2]= 147.760;
  ebxx_51[3]= 147.862; ebyy_51[3]= 146.034;
  ebxx_51[4]=   ebxx_51[0]; ebyy_51[4]=   ebyy_51[0];;
  _rzXtals[51] = new TPolyLine( 5, ebxx_51,ebyy_51);

  // eta=52
  double ebxx_52[5]; double ebyy_52[5];
  ebxx_52[0]= 135.480; ebyy_52[0]= 129.124;
  ebxx_52[1]= 133.780; ebyy_52[1]= 130.686;
  ebxx_52[2]= 149.350; ebyy_52[2]= 147.596;
  ebxx_52[3]= 151.253; ebyy_52[3]= 145.848;
  ebxx_52[4]=   ebxx_52[0]; ebyy_52[4]=   ebyy_52[0];;
  _rzXtals[52] = new TPolyLine( 5, ebxx_52,ebyy_52);

  // eta=53
  double ebxx_53[5]; double ebyy_53[5];
  ebxx_53[0]= 138.705; ebyy_53[0]= 129.124;
  ebxx_53[1]= 137.024; ebyy_53[1]= 130.706;
  ebxx_53[2]= 152.795; ebyy_53[2]= 147.429;
  ebxx_53[3]= 154.677; ebyy_53[3]= 145.659;
  ebxx_53[4]=   ebxx_53[0]; ebyy_53[4]=   ebyy_53[0];;
  _rzXtals[53] = new TPolyLine( 5, ebxx_53,ebyy_53);

  // eta=54
  double ebxx_54[5]; double ebyy_54[5];
  ebxx_54[0]= 141.968; ebyy_54[0]= 129.124;
  ebxx_54[1]= 140.306; ebyy_54[1]= 130.726;
  ebxx_54[2]= 156.275; ebyy_54[2]= 147.260;
  ebxx_54[3]= 158.135; ebyy_54[3]= 145.467;
  ebxx_54[4]=   ebxx_54[0]; ebyy_54[4]=   ebyy_54[0];;
  _rzXtals[54] = new TPolyLine( 5, ebxx_54,ebyy_54);

  // eta=55
  double ebxx_55[5]; double ebyy_55[5];
  ebxx_55[0]= 145.268; ebyy_55[0]= 129.124;
  ebxx_55[1]= 143.625; ebyy_55[1]= 130.746;
  ebxx_55[2]= 159.791; ebyy_55[2]= 147.089;
  ebxx_55[3]= 161.630; ebyy_55[3]= 145.274;
  ebxx_55[4]=   ebxx_55[0]; ebyy_55[4]=   ebyy_55[0];;
  _rzXtals[55] = new TPolyLine( 5, ebxx_55,ebyy_55);

  // eta=56
  double ebxx_56[5]; double ebyy_56[5];
  ebxx_56[0]= 148.631; ebyy_56[0]= 129.124;
  ebxx_56[1]= 146.997; ebyy_56[1]= 130.776;
  ebxx_56[2]= 163.357; ebyy_56[2]= 146.925;
  ebxx_56[3]= 165.172; ebyy_56[3]= 145.090;
  ebxx_56[4]=   ebxx_56[0]; ebyy_56[4]=   ebyy_56[0];;
  _rzXtals[56] = new TPolyLine( 5, ebxx_56,ebyy_56);

  // eta=57
  double ebxx_57[5]; double ebyy_57[5];
  ebxx_57[0]= 152.031; ebyy_57[0]= 129.124;
  ebxx_57[1]= 150.416; ebyy_57[1]= 130.794;
  ebxx_57[2]= 166.955; ebyy_57[2]= 146.760;
  ebxx_57[3]= 168.749; ebyy_57[3]= 144.905;
  ebxx_57[4]=   ebxx_57[0]; ebyy_57[4]=   ebyy_57[0];;
  _rzXtals[57] = new TPolyLine( 5, ebxx_57,ebyy_57);

  // eta=58
  double ebxx_58[5]; double ebyy_58[5];
  ebxx_58[0]= 155.471; ebyy_58[0]= 129.124;
  ebxx_58[1]= 153.874; ebyy_58[1]= 130.812;
  ebxx_58[2]= 170.591; ebyy_58[2]= 146.592;
  ebxx_58[3]= 172.364; ebyy_58[3]= 144.717;
  ebxx_58[4]=   ebxx_58[0]; ebyy_58[4]=   ebyy_58[0];;
  _rzXtals[58] = new TPolyLine( 5, ebxx_58,ebyy_58);

  // eta=59
  double ebxx_59[5]; double ebyy_59[5];
  ebxx_59[0]= 158.952; ebyy_59[0]= 129.124;
  ebxx_59[1]= 157.374; ebyy_59[1]= 130.829;
  ebxx_59[2]= 174.266; ebyy_59[2]= 146.422;
  ebxx_59[3]= 176.019; ebyy_59[3]= 144.528;
  ebxx_59[4]=   ebxx_59[0]; ebyy_59[4]=   ebyy_59[0];;
  _rzXtals[59] = new TPolyLine( 5, ebxx_59,ebyy_59);

  // eta=60
  double ebxx_60[5]; double ebyy_60[5];
  ebxx_60[0]= 162.476; ebyy_60[0]= 129.124;
  ebxx_60[1]= 160.917; ebyy_60[1]= 130.847;
  ebxx_60[2]= 177.982; ebyy_60[2]= 146.250;
  ebxx_60[3]= 179.714; ebyy_60[3]= 144.336;
  ebxx_60[4]=   ebxx_60[0]; ebyy_60[4]=   ebyy_60[0];;
  _rzXtals[60] = new TPolyLine( 5, ebxx_60,ebyy_60);

  // eta=61
  double ebxx_61[5]; double ebyy_61[5];
  ebxx_61[0]= 166.072; ebyy_61[0]= 129.124;
  ebxx_61[1]= 164.521; ebyy_61[1]= 130.877;
  ebxx_61[2]= 181.758; ebyy_61[2]= 146.090;
  ebxx_61[3]= 183.468; ebyy_61[3]= 144.156;
  ebxx_61[4]=   ebxx_61[0]; ebyy_61[4]=   ebyy_61[0];;
  _rzXtals[61] = new TPolyLine( 5, ebxx_61,ebyy_61);

  // eta=62
  double ebxx_62[5]; double ebyy_62[5];
  ebxx_62[0]= 169.710; ebyy_62[0]= 129.124;
  ebxx_62[1]= 168.178; ebyy_62[1]= 130.893;
  ebxx_62[2]= 185.572; ebyy_62[2]= 145.926;
  ebxx_62[3]= 187.262; ebyy_62[3]= 143.974;
  ebxx_62[4]=   ebxx_62[0]; ebyy_62[4]=   ebyy_62[0];;
  _rzXtals[62] = new TPolyLine( 5, ebxx_62,ebyy_62);

  // eta=63
  double ebxx_63[5]; double ebyy_63[5];
  ebxx_63[0]= 173.393; ebyy_63[0]= 129.124;
  ebxx_63[1]= 171.879; ebyy_63[1]= 130.909;
  ebxx_63[2]= 189.429; ebyy_63[2]= 145.759;
  ebxx_63[3]= 191.098; ebyy_63[3]= 143.790;
  ebxx_63[4]=   ebxx_63[0]; ebyy_63[4]=   ebyy_63[0];;
  _rzXtals[63] = new TPolyLine( 5, ebxx_63,ebyy_63);

  // eta=64
  double ebxx_64[5]; double ebyy_64[5];
  ebxx_64[0]= 177.121; ebyy_64[0]= 129.124;
  ebxx_64[1]= 175.626; ebyy_64[1]= 130.925;
  ebxx_64[2]= 193.330; ebyy_64[2]= 145.591;
  ebxx_64[3]= 194.979; ebyy_64[3]= 143.605;
  ebxx_64[4]=   ebxx_64[0]; ebyy_64[4]=   ebyy_64[0];;
  _rzXtals[64] = new TPolyLine( 5, ebxx_64,ebyy_64);

  // eta=65
  double ebxx_65[5]; double ebyy_65[5];
  ebxx_65[0]= 180.898; ebyy_65[0]= 129.124;
  ebxx_65[1]= 179.421; ebyy_65[1]= 130.940;
  ebxx_65[2]= 197.277; ebyy_65[2]= 145.422;
  ebxx_65[3]= 198.905; ebyy_65[3]= 143.418;
  ebxx_65[4]=   ebxx_65[0]; ebyy_65[4]=   ebyy_65[0];;
  _rzXtals[65] = new TPolyLine( 5, ebxx_65,ebyy_65);

  // eta=66
  double ebxx_66[5]; double ebyy_66[5];
  ebxx_66[0]= 185.940; ebyy_66[0]= 129.124;
  ebxx_66[1]= 184.468; ebyy_66[1]= 130.974;
  ebxx_66[2]= 202.474; ebyy_66[2]= 145.269;
  ebxx_66[3]= 204.086; ebyy_66[3]= 143.243;
  ebxx_66[4]=   ebxx_66[0]; ebyy_66[4]=   ebyy_66[0];;
  _rzXtals[66] = new TPolyLine( 5, ebxx_66,ebyy_66);

  // eta=67
  double ebxx_67[5]; double ebyy_67[5];
  ebxx_67[0]= 189.852; ebyy_67[0]= 129.124;
  ebxx_67[1]= 188.398; ebyy_67[1]= 130.989;
  ebxx_67[2]= 206.543; ebyy_67[2]= 145.107;
  ebxx_67[3]= 208.135; ebyy_67[3]= 143.065;
  ebxx_67[4]=   ebxx_67[0]; ebyy_67[4]=   ebyy_67[0];;
  _rzXtals[67] = new TPolyLine( 5, ebxx_67,ebyy_67);

  // eta=68
  double ebxx_68[5]; double ebyy_68[5];
  ebxx_68[0]= 193.814; ebyy_68[0]= 129.124;
  ebxx_68[1]= 192.378; ebyy_68[1]= 131.003;
  ebxx_68[2]= 210.660; ebyy_68[2]= 144.944;
  ebxx_68[3]= 212.232; ebyy_68[3]= 142.887;
  ebxx_68[4]=   ebxx_68[0]; ebyy_68[4]=   ebyy_68[0];;
  _rzXtals[68] = new TPolyLine( 5, ebxx_68,ebyy_68);

  // eta=69
  double ebxx_69[5]; double ebyy_69[5];
  ebxx_69[0]= 197.827; ebyy_69[0]= 129.124;
  ebxx_69[1]= 196.410; ebyy_69[1]= 131.017;
  ebxx_69[2]= 214.826; ebyy_69[2]= 144.780;
  ebxx_69[3]= 216.379; ebyy_69[3]= 142.707;
  ebxx_69[4]=   ebxx_69[0]; ebyy_69[4]=   ebyy_69[0];;
  _rzXtals[69] = new TPolyLine( 5, ebxx_69,ebyy_69);

  // eta=70
  double ebxx_70[5]; double ebyy_70[5];
  ebxx_70[0]= 201.894; ebyy_70[0]= 129.124;
  ebxx_70[1]= 200.495; ebyy_70[1]= 131.030;
  ebxx_70[2]= 219.045; ebyy_70[2]= 144.613;
  ebxx_70[3]= 220.576; ebyy_70[3]= 142.526;
  ebxx_70[4]=   ebxx_70[0]; ebyy_70[4]=   ebyy_70[0];;
  _rzXtals[70] = new TPolyLine( 5, ebxx_70,ebyy_70);

  // eta=71
  double ebxx_71[5]; double ebyy_71[5];
  ebxx_71[0]= 206.045; ebyy_71[0]= 129.124;
  ebxx_71[1]= 204.655; ebyy_71[1]= 131.057;
  ebxx_71[2]= 223.337; ebyy_71[2]= 144.459;
  ebxx_71[3]= 224.849; ebyy_71[3]= 142.356;
  ebxx_71[4]=   ebxx_71[0]; ebyy_71[4]=   ebyy_71[0];;
  _rzXtals[71] = new TPolyLine( 5, ebxx_71,ebyy_71);

  // eta=72
  double ebxx_72[5]; double ebyy_72[5];
  ebxx_72[0]= 210.248; ebyy_72[0]= 129.124;
  ebxx_72[1]= 208.876; ebyy_72[1]= 131.070;
  ebxx_72[2]= 227.678; ebyy_72[2]= 144.302;
  ebxx_72[3]= 229.171; ebyy_72[3]= 142.186;
  ebxx_72[4]=   ebxx_72[0]; ebyy_72[4]=   ebyy_72[0];;
  _rzXtals[72] = new TPolyLine( 5, ebxx_72,ebyy_72);

  // eta=73
  double ebxx_73[5]; double ebyy_73[5];
  ebxx_73[0]= 214.506; ebyy_73[0]= 129.124;
  ebxx_73[1]= 213.152; ebyy_73[1]= 131.082;
  ebxx_73[2]= 232.073; ebyy_73[2]= 144.144;
  ebxx_73[3]= 233.546; ebyy_73[3]= 142.014;
  ebxx_73[4]=   ebxx_73[0]; ebyy_73[4]=   ebyy_73[0];;
  _rzXtals[73] = new TPolyLine( 5, ebxx_73,ebyy_73);

  // eta=74
  double ebxx_74[5]; double ebyy_74[5];
  ebxx_74[0]= 218.821; ebyy_74[0]= 129.124;
  ebxx_74[1]= 217.484; ebyy_74[1]= 131.094;
  ebxx_74[2]= 236.523; ebyy_74[2]= 143.985;
  ebxx_74[3]= 237.977; ebyy_74[3]= 141.841;
  ebxx_74[4]=   ebxx_74[0]; ebyy_74[4]=   ebyy_74[0];;
  _rzXtals[74] = new TPolyLine( 5, ebxx_74,ebyy_74);

  // eta=75
  double ebxx_75[5]; double ebyy_75[5];
  ebxx_75[0]= 223.194; ebyy_75[0]= 129.124;
  ebxx_75[1]= 221.875; ebyy_75[1]= 131.106;
  ebxx_75[2]= 241.030; ebyy_75[2]= 143.824;
  ebxx_75[3]= 242.464; ebyy_75[3]= 141.668;
  ebxx_75[4]=   ebxx_75[0]; ebyy_75[4]=   ebyy_75[0];;
  _rzXtals[75] = new TPolyLine( 5, ebxx_75,ebyy_75);

  // eta=76
  double ebxx_76[5]; double ebyy_76[5];
  ebxx_76[0]= 227.679; ebyy_76[0]= 129.124;
  ebxx_76[1]= 226.367; ebyy_76[1]= 131.134;
  ebxx_76[2]= 245.637; ebyy_76[2]= 143.678;
  ebxx_76[3]= 247.054; ebyy_76[3]= 141.506;
  ebxx_76[4]=   ebxx_76[0]; ebyy_76[4]=   ebyy_76[0];;
  _rzXtals[76] = new TPolyLine( 5, ebxx_76,ebyy_76);

  // eta=77
  double ebxx_77[5]; double ebyy_77[5];
  ebxx_77[0]= 232.205; ebyy_77[0]= 129.124;
  ebxx_77[1]= 230.911; ebyy_77[1]= 131.145;
  ebxx_77[2]= 250.285; ebyy_77[2]= 143.527;
  ebxx_77[3]= 251.684; ebyy_77[3]= 141.343;
  ebxx_77[4]=   ebxx_77[0]; ebyy_77[4]=   ebyy_77[0];;
  _rzXtals[77] = new TPolyLine( 5, ebxx_77,ebyy_77);

  // eta=78
  double ebxx_78[5]; double ebyy_78[5];
  ebxx_78[0]= 236.792; ebyy_78[0]= 129.124;
  ebxx_78[1]= 235.515; ebyy_78[1]= 131.156;
  ebxx_78[2]= 254.992; ebyy_78[2]= 143.375;
  ebxx_78[3]= 256.372; ebyy_78[3]= 141.179;
  ebxx_78[4]=   ebxx_78[0]; ebyy_78[4]=   ebyy_78[0];;
  _rzXtals[78] = new TPolyLine( 5, ebxx_78,ebyy_78);

  // eta=79
  double ebxx_79[5]; double ebyy_79[5];
  ebxx_79[0]= 241.441; ebyy_79[0]= 129.124;
  ebxx_79[1]= 240.181; ebyy_79[1]= 131.166;
  ebxx_79[2]= 259.760; ebyy_79[2]= 143.222;
  ebxx_79[3]= 261.122; ebyy_79[3]= 141.015;
  ebxx_79[4]=   ebxx_79[0]; ebyy_79[4]=   ebyy_79[0];;
  _rzXtals[79] = new TPolyLine( 5, ebxx_79,ebyy_79);

  // eta=80
  double ebxx_80[5]; double ebyy_80[5];
  ebxx_80[0]= 246.154; ebyy_80[0]= 129.124;
  ebxx_80[1]= 244.911; ebyy_80[1]= 131.177;
  ebxx_80[2]= 264.591; ebyy_80[2]= 143.068;
  ebxx_80[3]= 265.934; ebyy_80[3]= 140.849;
  ebxx_80[4]=   ebxx_80[0]; ebyy_80[4]=   ebyy_80[0];;
  _rzXtals[80] = new TPolyLine( 5, ebxx_80,ebyy_80);

  // eta=81
  double ebxx_81[5]; double ebyy_81[5];
  ebxx_81[0]= 250.980; ebyy_81[0]= 129.124;
  ebxx_81[1]= 249.743; ebyy_81[1]= 131.206;
  ebxx_81[2]= 269.522; ebyy_81[2]= 142.932;
  ebxx_81[3]= 270.850; ebyy_81[3]= 140.696;
  ebxx_81[4]=   ebxx_81[0]; ebyy_81[4]=   ebyy_81[0];;
  _rzXtals[81] = new TPolyLine( 5, ebxx_81,ebyy_81);

  // eta=82
  double ebxx_82[5]; double ebyy_82[5];
  ebxx_82[0]= 255.868; ebyy_82[0]= 129.124;
  ebxx_82[1]= 254.647; ebyy_82[1]= 131.216;
  ebxx_82[2]= 274.516; ebyy_82[2]= 142.788;
  ebxx_82[3]= 275.827; ebyy_82[3]= 140.541;
  ebxx_82[4]=   ebxx_82[0]; ebyy_82[4]=   ebyy_82[0];;
  _rzXtals[82] = new TPolyLine( 5, ebxx_82,ebyy_82);

  // eta=83
  double ebxx_83[5]; double ebyy_83[5];
  ebxx_83[0]= 260.821; ebyy_83[0]= 129.124;
  ebxx_83[1]= 259.617; ebyy_83[1]= 131.225;
  ebxx_83[2]= 279.575; ebyy_83[2]= 142.643;
  ebxx_83[3]= 280.869; ebyy_83[3]= 140.386;
  ebxx_83[4]=   ebxx_83[0]; ebyy_83[4]=   ebyy_83[0];;
  _rzXtals[83] = new TPolyLine( 5, ebxx_83,ebyy_83);

  // eta=84
  double ebxx_84[5]; double ebyy_84[5];
  ebxx_84[0]= 265.843; ebyy_84[0]= 129.124;
  ebxx_84[1]= 264.655; ebyy_84[1]= 131.234;
  ebxx_84[2]= 284.702; ebyy_84[2]= 142.497;
  ebxx_84[3]= 285.978; ebyy_84[3]= 140.230;
  ebxx_84[4]=   ebxx_84[0]; ebyy_84[4]=   ebyy_84[0];;
  _rzXtals[84] = new TPolyLine( 5, ebxx_84,ebyy_84);

  // eta=85
  double ebxx_85[5]; double ebyy_85[5];
  ebxx_85[0]= 270.935; ebyy_85[0]= 129.124;
  ebxx_85[1]= 269.763; ebyy_85[1]= 131.244;
  ebxx_85[2]= 289.897; ebyy_85[2]= 142.350;
  ebxx_85[3]= 291.156; ebyy_85[3]= 140.074;
  ebxx_85[4]=   ebxx_85[0]; ebyy_85[4]=   ebyy_85[0];;
  _rzXtals[85] = new TPolyLine( 5, ebxx_85,ebyy_85);

  //  for( int ii=1; ii<=85; ii++ )
  //    registerTObject( _rzXtals[ii] );
}

void
MEEBDisplay::registerTObject( TObject* o )
{
  _list.push_back( o );
}

void
MEEBDisplay::refresh()
{
  for( list<TObject*>::iterator it=_list.begin(); 
       it!=_list.end(); ++it )
    {
      delete (*it);
      (*it) = 0;
    }
  _list.clear();
}
