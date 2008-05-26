// -*- C++ -*-
//
// Package:     Calo
// Class  :     ECalCaloTowerProxyRhoPhiZ2DBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: ECalCaloTowerProxyRhoPhiZ2DBuilder.cc,v 1.7 2008/03/13 03:02:01 chrjones Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TEveGeoShapeExtract.h"
#include "TGeoArb8.h"
#include "TEveManager.h"
#include "TH1F.h"
#include "TColor.h"
#include "TROOT.h"

#include <iostream>

// user include files
#include "Fireworks/Calo/interface/ECalCaloTowerProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWDisplayEvent.h"

#include "Fireworks/Core/interface/FWRhoPhiZView.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ECalCaloTowerProxyRhoPhiZ2DBuilder::ECalCaloTowerProxyRhoPhiZ2DBuilder()
{
}

// ECalCaloTowerProxyRhoPhiZ2DBuilder::ECalCaloTowerProxyRhoPhiZ2DBuilder(const ECalCaloTowerProxyRhoPhiZ2DBuilder& rhs)
// {
//    // do actual copying here;
// }

ECalCaloTowerProxyRhoPhiZ2DBuilder::~ECalCaloTowerProxyRhoPhiZ2DBuilder()
{
}

//
// member functions
//
void 
ECalCaloTowerProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem,
					    TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"ECAL RhoPhi",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }
   
   const CaloTowerCollection* towers=0;
   iItem->get(towers);
   if(0==towers) {
      std::cout <<"Failed to get CaloTowers"<<std::endl;
      return;
   }
   // double eta_limit = 1.5;
   // if ( m_parameters.GetBoolParameter("ShowEndCaps") ) eta_limit = 1000;
   tList->AddElement( TEveGeoShape::ImportShapeExtract( getRhoPhiElements("towers",
									  towers, 
									  iItem->defaultDisplayProperties().color(), 
									  false,
									  1.5,
									  FWDisplayEvent::getCaloScale())
							, 0 ) );
}

void 
ECalCaloTowerProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem,
					    TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"ECAL RhoZ",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }
   
   const CaloTowerCollection* towers=0;
   iItem->get(towers);
   if(0==towers) {
      std::cout <<"Failed to get CaloTowers"<<std::endl;
      return;
   }
   tList->AddElement( TEveGeoShape::ImportShapeExtract( getRhoZElements("towers", 
									towers,
									iItem->defaultDisplayProperties().color(), 
									false,
									FWDisplayEvent::getCaloScale()),
							0 ) );
}


TEveGeoShapeExtract*
ECalCaloTowerProxyRhoPhiZ2DBuilder::getRhoPhiElements(const char* name, 
						      const CaloTowerCollection* towers, 
						      Int_t color, 
						      bool hcal,
						      double eta_limit /*= 1.5*/,
						      double scale /*= 2*/ )
{
   
   // NOTE:
   //       Here we assume 72 bins in phi. At high eta we have only 36 and at the
   //       very end 18 bins. These large bins are splited among smaller bins 
   //       decreasing energy in each entry by factor of 2 and 4 for 36 and 18 bin 
   //       cases. Other options will be implemented later
   // 
   // http://ecal-od-software.web.cern.ch/ecal-od-software/documents/documents/cal_newedm_roadmap_v1_0.pdf
   // Eta mapping:
   //   ieta - [-41,-1]+[1,41] - total 82 bins 
   //   calo tower gives eta of the ceneter of each bin
   //   size:
   //      0.087 - [-20,-1]+[1,20]
   //      the rest have variable size from 0.09-0.30
   // Phi mapping:
   //   iphi - [1-72]
   //   calo tower gives phi of the center of each bin
   //   for |ieta|<=20 phi bins are all of the same size
   //      iphi 36-37 transition corresponds to 3.1 -> -3.1 transition
   //   for 20 < |ieta| < 40
   //      there are only 36 active bins corresponding to odd numbers
   //      iphi 35->37, corresponds to 3.05 -> -3.05 transition
   //   for |ieta| >= 40
   //      there are only 18 active bins 3,7,11,15 etc
   //      iphi 31 -> 35, corresponds to 2.79253 -> -3.14159 transition
   std::vector<double> h_ecal(72,0);
   std::vector<double> h_hcal(72,0);
   for(CaloTowerCollection::const_iterator tower = towers->begin(); tower != towers->end(); ++tower)
     {
	if ( fabs(tower->eta()) > eta_limit ) continue;
	// looks like phi of CaloTower points to the edge, lets shift phi a bit
	// iphi = 0 corresponds to phi = -Pi, which points in -x direction. 
	int iphi = int ( (tower->phi()/2/M_PI+0.5)*72 - 0.1);
	assert ( iphi>=0 && iphi<72 );

	if ( tower->id().ietaAbs() <= 20 ) {
	   h_ecal[iphi] += tower->emEt();
	   h_hcal[iphi] += tower->hadEt();
	}
	
	if ( tower->id().ietaAbs() > 20 && tower->id().ietaAbs()<40 ) {
	   assert ( iphi+1<72 );
	   h_ecal[iphi]   += tower->emEt()/2;
	   h_ecal[iphi+1] += tower->emEt()/2;
	   h_hcal[iphi]   += tower->hadEt()/2;
	   h_hcal[iphi+1] += tower->hadEt()/2;
	}
	
	if ( tower->id().ietaAbs() >= 40 )  {
	   assert ( iphi+3<72 );
	   h_ecal[iphi]   += tower->emEt()/4;
	   h_ecal[iphi+1] += tower->emEt()/4;
	   h_ecal[iphi+2] += tower->emEt()/4;
	   h_ecal[iphi+3] += tower->emEt()/4;
	   h_hcal[iphi]   += tower->hadEt()/4;
	   h_hcal[iphi+1] += tower->hadEt()/4;
	   h_hcal[iphi+2] += tower->hadEt()/4;
	   h_hcal[iphi+3] += tower->hadEt()/4;
	}
     }
   if (scale < 0 ) {
      // auto scale mode
      double maxValue = 1;
      for ( unsigned int i=0; i<72; ++i ) {
	 if ( maxValue < h_ecal[i]+h_hcal[i] ) maxValue = h_ecal[i]+h_hcal[i];
      }
      scale = 200 / maxValue;
      printf("Final scale value for rho-phi: %f\n", scale);
   }
   
   // Make objects representing towers
   TEveGeoShapeExtract* container = new TEveGeoShapeExtract( name );
   double offset = M_PI; // need Pi rotation in phi to get proper position
   double r_ecal = 129;
   for ( unsigned int i=0; i<72; ++i ) {
      double r = r_ecal;
      if ( hcal ) r +=scale*h_ecal[i];
      double size(0);
      if (hcal) 
	size = scale*h_hcal[i]/2;
      else
	size = scale*h_ecal[i]/2;
      double r_trap_min = r*M_PI/36;
      double r_trap_max = (r+size*2)*M_PI/36;
      TEveTrans t;
      t(1,1) = 0; t(1,2) = 0; t(1,3) = 1;
      t(2,1) = 1; t(2,2) = 0; t(2,3) = 0;
      t(3,1) = 0; t(3,2) = 1; t(3,3) = 0;
      t(1,4) = r+size; t(2,4) = 0; t(3,4) = 0;
      t.RotatePF(1,2,offset+(i+0.5)*M_PI/36);
      std::ostringstream s;
      s << "Phi" << i;
      TEveGeoShapeExtract* extract = new TEveGeoShapeExtract(s.str().c_str());
      extract->SetTrans(t.Array());
      
      TColor* c = gROOT->GetColor(color);
      Float_t rgba[4] = { 1, 0, 0, 1 };
      if (c) {
	 rgba[0] = c->GetRed();
	 rgba[1] = c->GetGreen();
	 rgba[2] = c->GetBlue();
      }
   
      extract->SetRGBA(rgba);
      extract->SetRnrSelf(kTRUE);
      extract->SetRnrElements(kTRUE);
      extract->SetShape( new TGeoTrap(size,0,0,0.001,r_trap_min/2,r_trap_min/2,0,0.001,r_trap_max/2,r_trap_max/2,0) );
      container->AddElement( extract );
   }
   return container;
}

std::vector<std::pair<double,double> > 
  ECalCaloTowerProxyRhoPhiZ2DBuilder::getThetaBins()
{
   std::vector<std::pair<double,double> > thetaBins(82);
   for ( unsigned int i = 0; i < 82; ++i )
     {
	thetaBins[i].first  = 2*atan( exp(-fw3dlego::xbins[i]) );
	thetaBins[i].second = 2*atan( exp(-fw3dlego::xbins[i+1]) );
     }
   return thetaBins;
}

TEveGeoShapeExtract*
ECalCaloTowerProxyRhoPhiZ2DBuilder::getRhoZElements(const char* name, 
						    double size,
						    double r,
						    double theta,
						    double dTheta, 
						    Int_t color,
						    bool top)
{
   double r_trap_min = r*dTheta;
   double r_trap_max = (r+size*2)*dTheta;
   // std::cout << "theta: " << theta << " \tdTheta: " << dTheta << " \tr: " << r <<
   // " \t size: " << size << " \t r_trap_min: " << r_trap_min << " \tr_trap_max: " <<
   // r_trap_max << std::endl;
   TEveTrans t;
   t(1,1) = 1; t(1,2) = 0; t(1,3) = 0;
   t(2,1) = 0; t(2,2) = 1; t(2,3) = 0;
   t(3,1) = 0; t(3,2) = 0; t(3,3) = 1;
   t(1,4) = 0; t(2,4) = r+size; t(3,4) = 0;
   t.RotateLF(2,3,-M_PI/2);
   t.RotateLF(1,2,M_PI/2);
   if ( top ) 
     t.RotatePF(2,3,M_PI/2-theta);
   else {
      t.RotatePF(1,2,M_PI);
      t.RotatePF(2,3,-M_PI/2+theta);
   }
   
   TEveGeoShapeExtract* extract = new TEveGeoShapeExtract(name);
   extract->SetTrans(t.Array());
   TColor* c = gROOT->GetColor(color);
   Float_t rgba[4] = { 1, 0, 0, 1 };
   if (c) {
      rgba[0] = c->GetRed();
      rgba[1] = c->GetGreen();
      rgba[2] = c->GetBlue();
   }
   
   extract->SetRGBA(rgba);
   extract->SetRnrSelf(kTRUE);
   extract->SetRnrElements(kTRUE);
   extract->SetShape( new TGeoTrap(size,0,0,0.001,r_trap_min/2,r_trap_min/2,0,0.001,r_trap_max/2,r_trap_max/2,0) );
   return extract;
}

TEveGeoShapeExtract*
ECalCaloTowerProxyRhoPhiZ2DBuilder::getRhoZElements(const char* name, 
						    const CaloTowerCollection* towers, 
						    Int_t color, 
						    bool hcal,
						    double scale /*= 2*/)
{
   // NOTE:
   //      We derive eta bin size from xbins array used for LEGO assuming that all 82
   //      eta bins are accounted there. 
   assert ( sizeof(fw3dlego::xbins)/sizeof(*fw3dlego::xbins) == 82+1 );
   static const std::vector<std::pair<double,double> > thetaBins = getThetaBins();
   
   std::vector<double> h_ecal_top(82,0);
   std::vector<double> h_hcal_top(82,0);
   std::vector<double> h_ecal_bottom(82,0);
   std::vector<double> h_hcal_bottom(82,0);
   for(CaloTowerCollection::const_iterator tower = towers->begin(); tower != towers->end(); ++tower)
     {
	unsigned int ieta = 41 + tower->id().ieta();
	if ( ieta > 40 ) --ieta;
	assert( ieta <= 82 );
	if ( tower->phi()>0 ) {
	   h_ecal_top[ieta] += tower->emEt();
	   h_hcal_top[ieta] += tower->hadEt();
	} else {
	   h_ecal_bottom[ieta] += tower->emEt();
	   h_hcal_bottom[ieta] += tower->hadEt();
	}
     }

   if (scale < 0 ) {
      // auto scale mode
      double maxValue = 1;
      for ( unsigned int i=0; i<72; ++i ) {
	 if ( maxValue < h_ecal_top[i]+h_hcal_top[i] ) maxValue = h_ecal_top[i]+h_hcal_top[i];
	 if ( maxValue < h_ecal_bottom[i]+h_hcal_bottom[i] ) maxValue = h_ecal_bottom[i]+h_hcal_bottom[i];
      }
      scale = 200 / maxValue;
      printf("Final scale value for rho-z: %f\n", scale);
   }
   
   // Make objects representing towers
   TEveGeoShapeExtract* container = new TEveGeoShapeExtract( name );
   double z_ecal = 310; // ECAL endcap inner surface
   double r_ecal = 134;
   double transition_angle = atan(r_ecal/z_ecal);
   for ( unsigned int i=0; i<82; ++i ) {
      double theta = ( thetaBins[i].first + thetaBins[i].second )/2;
      double dTheta = fabs( thetaBins[i].first - thetaBins[i].second );
      double r(0); // distance from the origin
      if ( theta < transition_angle || M_PI-theta < transition_angle )
	r = z_ecal/fabs(cos(theta));
      else
	r = r_ecal/sin(theta);
      
      std::ostringstream s;
      s << "Eta" << i;
      
      if ( hcal ) {
	 container->AddElement( getRhoZElements( ("Top"+s.str()).c_str(), scale*h_hcal_top[i]/2, r + scale*h_ecal_top[i], theta, dTheta, color, true ) );
	 container->AddElement( getRhoZElements( ("Bottom"+s.str()).c_str(), scale*h_hcal_bottom[i]/2, r + scale*h_ecal_bottom[i], theta, dTheta, color, false ) );
      } else {
	 container->AddElement( getRhoZElements( ("Top"+s.str()).c_str(), scale*h_ecal_top[i]/2, r, theta, dTheta, color, true ) );
	 container->AddElement( getRhoZElements( ("Bottom"+s.str()).c_str(), scale*h_ecal_bottom[i]/2, r, theta, dTheta, color, false ) );
      }
   }
   return container;
}


//
// const member functions
//

//
// static member functions
//
