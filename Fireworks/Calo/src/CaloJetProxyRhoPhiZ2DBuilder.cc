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
// $Id: CaloJetProxyRhoPhiZ2DBuilder.cc,v 1.1 2008/02/18 10:54:55 dmytro Exp $
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
#include "TEvePointSet.h"
#include "TEveStraightLineSet.h"

// user include files
#include "Fireworks/Calo/interface/CaloJetProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Calo/interface/ECalCaloTowerProxyRhoPhiZ2DBuilder.h"

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
   
   double r_ecal = 129;
   double scale = 2;
   
   unsigned int i = 0;
   for(reco::CaloJetCollection::const_iterator jet = jets->begin(); jet != jets->end(); ++jet) {

      TEveTrans t;
      t(1,1) = 1; t(1,2) = 0; t(1,3) = 0;
      t(2,1) = 0; t(2,2) = 1; t(2,3) = 0;
      t(3,1) = 0; t(3,2) = 0; t(3,3) = 1;
      t(1,4) = 0; t(2,4) = 0; t(3,4) = 0;
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

      std::pair<double,double> phiRange = getPhiRange( *jet );
      double min_phi = phiRange.first-M_PI/36/2;
      double max_phi = phiRange.second+M_PI/36/2;
      
      double phi = jet->phi();
      double dPhi1 = max_phi - phi;
      double dPhi2 = phi - min_phi;

      extract->SetRGBA(rgba);
      extract->SetRnrSelf(kTRUE);
      extract->SetRnrElements(kTRUE);
      
      double size = scale*jet->et();
      Double_t points[16];
      
      // define outer edge
      double rho = (r_ecal+size)/cos(dPhi1);
      points[2] = rho*cos(max_phi);
      points[3] = rho*sin(max_phi);
      rho = (r_ecal+size)/cos(dPhi2);
      points[4] = rho*cos(min_phi);
      points[5] = rho*sin(min_phi);
      rho = r_ecal/cos(dPhi1);
      points[0] = rho*cos(max_phi);
      points[1] = rho*sin(max_phi);
      rho = r_ecal/cos(dPhi2);
      points[6] = rho*cos(min_phi);
      points[7] = rho*sin(min_phi);
      
      for( int i = 0; i<8; ++i ) points[i+8] = points[i];
      extract->SetShape( new TGeoArb8(0,points) );
      TEveElement* element = TEveGeoShape::ImportShapeExtract( extract, 0 );
      element->SetMainTransparency(90);
      
      /*
      TEveStraightLineSet* marker = new TEveStraightLineSet("jet centroid");
      marker->SetLineWidth(4);
      marker->SetLineColor(  tList->GetMainColor() );
      marker->AddLine(0., (jet->phi()>0 ? (r-5)*fabs(sin(theta)) : -(r-5)*fabs(sin(theta))), (r-5)*cos(theta),
		      0., (jet->phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta) );
      element->AddElement(marker);
       */
      
      TEvePointSet *marker2 = new TEvePointSet("jet centroid", 1);
      marker2->SetNextPoint(r_ecal*cos(phi),r_ecal*sin(phi),0);
      marker2->SetMarkerSize(2);
      marker2->SetMarkerStyle(20);
      marker2->SetMarkerColor(tList->GetMainColor());
      element->AddElement(marker2);
      
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
   
   // NOTE:
   //      We derive eta bin size from xbins array used for LEGO assuming that all 82
   //      eta bins are accounted there. 
   assert ( sizeof(fw3dlego::xbins)/sizeof(*fw3dlego::xbins) == 82+1 );
   static const std::vector<std::pair<double,double> > thetaBins = ECalCaloTowerProxyRhoPhiZ2DBuilder::getThetaBins();

   
   double scale = 2;
   double z_ecal = 304.5; // ECAL endcap inner surface
   double r_ecal = 129;
   double transition_angle = atan(r_ecal/z_ecal);
   
   unsigned int i = 0;
   for(reco::CaloJetCollection::const_iterator jet = jets->begin(); jet != jets->end(); ++jet) {
      std::pair<int,int> iEtaRange = getiEtaRange( *jet );
      
      TEveTrans t;
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
      double max_theta = thetaBins[iEtaRange.first].first;
      double min_theta = thetaBins[iEtaRange.second].second;;
      
      double theta = jet->theta();
      double dTheta1 = max_theta - theta;
      double dTheta2 = theta - min_theta;
      
      // distance from the origin of the jet centroid
      // energy is measured from this point
      // if jet is made of a single tower, the length of the jet will 
      // be identical to legth of the displayed tower
      double r(0); 
      if ( theta < transition_angle || M_PI-theta < transition_angle )
	r = z_ecal/fabs(cos(theta));
      else
	r = r_ecal/sin(theta);
      
      double size = scale*jet->et();
      
      // Shape:
      //   energy - distance from the center of inner most edge to the 
      //            center of the outer most edge
      Double_t points[16];
      
      // define outer edge
      double rho = (r+size)/cos(dTheta1);
      points[2] = rho*cos(max_theta);
      points[3] = jet->phi()>0 ? rho*sin(max_theta) : -rho*sin(max_theta);
      rho = (r+size)/cos(dTheta2);
      points[4] = rho*cos(min_theta);
      points[5] = jet->phi()>0 ? rho*sin(min_theta) : -rho*sin(min_theta);
	
/*      // inner edge:
      // - horizontal if jet is contained in the barrel
      // - vertical if jet is contained in the endcaps
      // - parallel to the outer edge if in the transition area
      
      if ( max_theta < M_PI - transition_angle && min_theta > transition_angle )
	{
	   points[0] = r_ecal/tan(max_theta);
	   points[1] = jet->phi()>0 ? r_ecal : -r_ecal;
	   points[6] = r_ecal/tan(min_theta);
	   points[7] = jet->phi()>0 ? r_ecal : -r_ecal;
	}
      
      if ( min_theta > M_PI - transition_angle )
	{
	   points[0] = -z_ecal;
	   points[1] = jet->phi()>0 ? z_ecal*fabs(tan(max_theta)) : -z_ecal*fabs(tan(max_theta));
	   points[6] = -z_ecal;
	   points[7] = jet->phi()>0 ? z_ecal*fabs(tan(min_theta)) : -z_ecal*fabs(tan(min_theta));
	}
      
      if ( max_theta < transition_angle )
	{
	   points[0] = z_ecal;
	   points[1] = jet->phi()>0 ? z_ecal*fabs(tan(max_theta)) : -z_ecal*fabs(tan(max_theta));
	   points[6] = z_ecal;
	   points[7] = jet->phi()>0 ? z_ecal*fabs(tan(min_theta)) : -z_ecal*fabs(tan(min_theta));
	}

      if ( ( min_theta < transition_angle && max_theta > transition_angle) ||
	   ( min_theta < M_PI - transition_angle && max_theta > M_PI - transition_angle ) )
	{
*/	   rho = r/cos(dTheta1);
	   points[0] = rho*cos(max_theta);
	   points[1] = jet->phi()>0 ? rho*sin(max_theta) : -rho*sin(max_theta);
	   rho = r/cos(dTheta2);
	   points[6] = rho*cos(min_theta);
	   points[7] = jet->phi()>0 ? rho*sin(min_theta) : -rho*sin(min_theta);
//	}

//      printf("Jet (et,theta,phi): (%0.2f,%0.2f, %0.2f), angles (min,max,trans): ((%0.2f,%0.2f, %0.2f)\n",
//	     jet->et(), jet->theta(), jet->phi(), min_theta, max_theta, transition_angle);
      for( int i = 0; i<8; ++i ) points[i+8] = points[i];
      extract->SetShape( new TGeoArb8(0,points) );
      TEveElement* element = TEveGeoShape::ImportShapeExtract( extract, 0 );
      element->SetMainTransparency(90);
      
      /*
      TEveStraightLineSet* marker = new TEveStraightLineSet("jet centroid");
      marker->SetLineWidth(4);
      marker->SetLineColor(  tList->GetMainColor() );
      marker->AddLine(0., (jet->phi()>0 ? (r-5)*fabs(sin(theta)) : -(r-5)*fabs(sin(theta))), (r-5)*cos(theta),
		      0., (jet->phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta) );
      element->AddElement(marker);
       */
      
      TEvePointSet *marker2 = new TEvePointSet("jet centroid", 1);
      marker2->SetNextPoint(0, (jet->phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta));
      marker2->SetMarkerSize(2);
      marker2->SetMarkerStyle(20);
      marker2->SetMarkerColor(tList->GetMainColor());
      element->AddElement(marker2);
      
      tList->AddElement( element );
   }
}
   
   
std::pair<int,int> CaloJetProxyRhoPhiZ2DBuilder::getiEtaRange( const reco::CaloJet& jet )
{
   int min =  100;
   int max = -100;

   std::vector<CaloTowerRef> towers = jet.getConstituents();
   for ( std::vector<CaloTowerRef>::const_iterator tower = towers.begin(); 
	 tower != towers.end(); ++tower ) 
     {
	unsigned int ieta = 41 + (*tower)->id().ieta();
	if ( ieta > 40 ) --ieta;
	assert( ieta <= 82 );
	
	if ( int(ieta) > max ) max = ieta;
	if ( int(ieta) < min ) min = ieta;
     }
   if ( min > max ) return std::pair<int,int>(0,0);
   return std::pair<int,int>(min,max);
}

std::pair<double,double> CaloJetProxyRhoPhiZ2DBuilder::getPhiRange( const reco::CaloJet& jet )
{
   double min =  100;
   double max = -100;

   std::vector<CaloTowerRef> towers = jet.getConstituents();
   for ( std::vector<CaloTowerRef>::const_iterator tower = towers.begin(); 
	 tower != towers.end(); ++tower ) 
     {
	double phi = (*tower)->phi();
	// make phi continuous around jet phi
	if ( phi - jet.phi() > M_PI ) phi -= 2*M_PI;
	if ( jet.phi() - phi > M_PI ) phi += 2*M_PI;
	if ( phi > max ) max = phi;
	if ( phi < min ) min = phi;
     }
   
   if ( min > max ) return std::pair<double,double>(0,0);
   
   return std::pair<double,double>(min,max);
}
