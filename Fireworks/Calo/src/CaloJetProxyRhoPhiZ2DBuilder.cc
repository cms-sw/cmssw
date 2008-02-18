// -*- C++ -*-
//
// Package:     Calo
// Class  :     CaloJetProxyRhoPhiZ2DBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: CaloJetProxyRhoPhiZ2DBuilder.cc,v 1.1 2008/02/03 02:57:10 dmytro Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TEveGeoShapeExtract.h"
#include "TGeoArb8.h"
#include "TEveManager.h"
#include "TGeoSphere.h"
#include "TGeoTube.h"
#include "TH1F.h"
#include "TColor.h"
#include "TROOT.h"

// user include files
#include "Fireworks/Calo/interface/CaloJetProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"

#include "DataFormats/JetReco/interface/CaloJetfwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CaloJetProxyRhoPhiZ2DBuilder::CaloJetProxyRhoPhiZ2DBuilder()
{
}

// CaloJetProxyRhoPhiZ2DBuilder::CaloJetProxyRhoPhiZ2DBuilder(const CaloJetProxyRhoPhiZ2DBuilder& rhs)
// {
//    // do actual copying here;
// }

CaloJetProxyRhoPhiZ2DBuilder::~CaloJetProxyRhoPhiZ2DBuilder()
{
}

//
// member functions
//
void 
CaloJetProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem,
					    TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"Jets RhoPhi",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }
   
   const reco::CaloJetCollection* jets=0;
   iItem->get(jets);
   if(0==jets) {
      std::cout <<"Failed to get CaloJets"<<std::endl;
      return;
   }
   
   double minJetEt = 15; // less energetic jets will be invisible
   // double z_ecal = 304.5; // ECAL endcap inner surface
   double r = 128;
   double jetPhiSize = 0.5;
   // double transition_angle = atan(r/z_ecal);
   // double r_trap_min = 0;
   // double r_trap_max = r*jetPhiSize;
   double offset = 0;
   
   unsigned int i = 0;
   for(reco::CaloJetCollection::const_iterator jet = jets->begin(); jet != jets->end(); ++jet) {
      if (jet->pt()<minJetEt) continue;
      TEveTrans t;
      // t(1,1) = 0; t(1,2) = 0; t(1,3) = 1;
      // t(2,1) = 1; t(2,2) = 0; t(2,3) = 0;
      // t(3,1) = 0; t(3,2) = 1; t(3,3) = 0;
      // t(1,4) = r/2; t(2,4) = 0; t(3,4) = 0;
      // t.RotatePF(1,2, offset+jet->phi() );
      t(1,1) = 1; t(1,2) = 0; t(1,3) = 1;
      t(2,1) = 0; t(2,2) = 1; t(2,3) = 0;
      t(3,1) = 0; t(3,2) = 0; t(3,3) = 1;
      t(1,4) = 0; t(2,4) = 0; t(3,4) = 0;
      t.RotatePF(1,2, offset+jet->phi() );
      std::ostringstream s;
      s << "Jet " << i;
      ++i;
      TEveGeoShapeExtract* extract = new TEveGeoShapeExtract(s.str().c_str());
      extract->SetTrans(t.Array());
      
      TColor* c = gROOT->GetColor( tList->GetMainColor() );
      Float_t rgba[4] = { 0.1, 0.1, 0.1, 1 };
      if (c) {
	 rgba[0] = c->GetRed();
	 rgba[1] = c->GetGreen();
	 rgba[2] = c->GetBlue();
      }
      
      extract->SetRGBA(rgba);
      extract->SetRnrSelf(kTRUE);
      extract->SetRnrElements(kTRUE);
      // extract->SetShape( new TGeoTrap(r/2,0,0,1,r_trap_min/2,r_trap_min/2,0,1,r_trap_max/2,r_trap_max/2,0) );
      TGeoSphere* shape = new TGeoSphere(double(0),r, 89.99, 90.01, -180/M_PI*jetPhiSize/2, 180/M_PI*jetPhiSize/2 );
      shape->SetNumberOfDivisions(1);
      extract->SetShape( shape );
      // extract->SetShape( new TGeoTubeSeg(double(0),r,1,-180/M_PI*jetPhiSize/2, 180/M_PI*jetPhiSize/2) );
      TEveElement* element = TEveGeoShape::ImportShapeExtract( extract, 0 );
      element->SetMainTransparency(90);
      tList->AddElement( element );
   }
}

void 
CaloJetProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem,
					    TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"Jets RhoZ",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }
   
   const reco::CaloJetCollection* jets=0;
   iItem->get(jets);
   if(0==jets) {
      std::cout <<"Failed to get CaloJets"<<std::endl;
      return;
   }
   double minJetEt = 15; // less energetic jets will be invisible
   double z = 300; // ECAL endcap inner surface
   double r = 120;
   double jetSize = 0.5;
   double transition_angle = atan(r/z);
   // double r_trap_min = 0;
   // double r_trap_max = r*jetPhiSize;
   // double offset = 0;
   
   unsigned int i = 0;
   for(reco::CaloJetCollection::const_iterator jet = jets->begin(); jet != jets->end(); ++jet) {
      if (jet->pt()<minJetEt) continue;
      TEveTrans t;
      // t(1,1) = 0; t(1,2) = 0; t(1,3) = 1;
      // t(2,1) = 1; t(2,2) = 0; t(2,3) = 0;
      // t(3,1) = 0; t(3,2) = 1; t(3,3) = 0;
      // t(1,4) = r/2; t(2,4) = 0; t(3,4) = 0;
      // t.RotatePF(1,2, offset+jet->phi() );
      t(1,1) = 1; t(1,2) = 0; t(1,3) = 0;
      t(2,1) = 0; t(2,2) = 1; t(2,3) = 0;
      t(3,1) = 0; t(3,2) = 0; t(3,3) = 1;
      t(1,4) = 0; t(2,4) = 0; t(3,4) = 0;
      t.RotateLF(1,3,M_PI/2 );
      std::ostringstream s;
      s << "Jet " << i;
      ++i;
      TEveGeoShapeExtract* extract = new TEveGeoShapeExtract(s.str().c_str());
      extract->SetTrans(t.Array());
      
      TColor* c = gROOT->GetColor( tList->GetMainColor() );
      Float_t rgba[4] = { 0.1, 0.1, 0.1, 1 };
      if (c) {
	 rgba[0] = c->GetRed();
	 rgba[1] = c->GetGreen();
	 rgba[2] = c->GetBlue();
      }
      
      extract->SetRGBA(rgba);
      extract->SetRnrSelf(kTRUE);
      extract->SetRnrElements(kTRUE);
      double min_theta = getTheta(jet->eta()+jetSize);
      double max_theta = getTheta(jet->eta()-jetSize);
      Double_t points[16];
      points[0] = 0;
      points[1] = 0;
      
      if ( max_theta > M_PI - transition_angle )
	{
	   points[6] = -z;
	   points[7] = jet->phi()>0 ? z*fabs(tan(max_theta)) : -z*fabs(tan(max_theta));
	} else 
	if ( max_theta < transition_angle ) {
	   points[6] = z;
	   points[7] = jet->phi()>0 ? z*fabs(tan(max_theta)) : -z*fabs(tan(max_theta));
	} else {
	   points[6] = r/tan(max_theta);
	   points[7] = jet->phi()>0 ? r : -r;
	}
      
      if ( min_theta > M_PI - transition_angle )
	{
	   points[2] = -z;
	   points[3] = jet->phi()>0 ? z*fabs(tan(min_theta)) : -z*fabs(tan(min_theta));
	} else 
	if ( min_theta < transition_angle ) {
	   points[2] = z;
	   points[3] = jet->phi()>0 ? z*fabs(tan(min_theta)) : -z*fabs(tan(min_theta));
	} else {
	   points[2] = r/tan(min_theta);
	   points[3] = jet->phi()>0 ? r : -r;
	}
      
      if ( min_theta < M_PI - transition_angle && max_theta > M_PI - transition_angle )
	{
	   points[4] = -z;
	   points[5] = jet->phi()>0 ? r : -r;
	} else
	if ( min_theta < transition_angle && max_theta > transition_angle ) {
	   points[4] = z;
	   points[5] = jet->phi()>0 ? r : -r;
	} else {
	   points[4] = points[2];
	   points[5] = points[3];
	}
      
      std::cout << "Jet theta: " << jet->theta() << std::endl;
      for( int i = 0; i<8; ++i ){
	 points[i+8] = points[i];
	 std::cout << "\t" << points[i];
      }
      std::cout << std::endl;
      extract->SetShape( new TGeoArb8(0,points) );
      TEveElement* element = TEveGeoShape::ImportShapeExtract( extract, 0 );
      element->SetMainTransparency(90);
      tList->AddElement( element );
   }
}
   
   
