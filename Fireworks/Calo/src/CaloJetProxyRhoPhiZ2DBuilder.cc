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
// $Id: CaloJetProxyRhoPhiZ2DBuilder.cc,v 1.7 2008/05/12 15:38:00 dmytro Exp $
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
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "Fireworks/Core/interface/FWDisplayEvent.h"

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
   
   double r_ecal = 126;
   double scale = FWDisplayEvent::getCaloScale();
   if ( scale < 0 ) scale = 2;
   //double minJetEt = 15;
   double minJetEt = 0;
   
   fw::NamedCounter counter("jet");

   for(reco::CaloJetCollection::const_iterator jet = jets->begin(); 
       jet != jets->end(); ++jet, ++counter) {
      TEveElementList* container = new TEveElementList( counter.str().c_str() );
      std::pair<double,double> phiRange = getPhiRange( *jet );
      double min_phi = phiRange.first-M_PI/36/2;
      double max_phi = phiRange.second+M_PI/36/2;
      
      double phi = jet->phi();

      double size = scale*jet->et();
      TGeoBBox *sc_box = new TGeoTubeSeg(r_ecal - 1, r_ecal + 1, 1, min_phi * 180 / M_PI, max_phi * 180 / M_PI);
      TEveGeoShapeExtract *sc = fw::getShapeExtract( "spread", sc_box, iItem->defaultDisplayProperties().color() );
      
      if ( jet->et() > minJetEt ) {
	 TEveStraightLineSet* marker = new TEveStraightLineSet("energy");
	 marker->SetLineWidth(4);
	 marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
	 TEveElement* element = TEveGeoShape::ImportShapeExtract(sc, 0);
	 element->SetPickable(kTRUE);
	 container->AddElement(element);
	 marker->AddLine( r_ecal*cos(phi), r_ecal*sin(phi), 0, (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
	 container->AddElement(marker);
      }
      tList->AddElement(container);
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

   
   double scale = FWDisplayEvent::getCaloScale();
   if ( scale < 0 ) scale = 2;
   double z_ecal = 306; // ECAL endcap inner surface
   double r_ecal = 126;
   double transition_angle = atan(r_ecal/z_ecal);
   //double minJetEt = 15;
   double minJetEt = 0;
   fw::NamedCounter counter("jet");

   for(reco::CaloJetCollection::const_iterator jet = jets->begin(); 
       jet != jets->end(); ++jet, ++counter) {
      TEveElementList* container = new TEveElementList( counter.str().c_str() );
      std::pair<int,int> iEtaRange = getiEtaRange( *jet );
      
      double max_theta = thetaBins[iEtaRange.first].first;
      double min_theta = thetaBins[iEtaRange.second].second;;
      
      double theta = jet->theta();
      
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
      
      if ( jet->et() > minJetEt ) {
	 TEveStraightLineSet* marker = new TEveStraightLineSet("energy");
	 marker->SetLineWidth(4);
	 marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
	 marker->AddLine(0., (jet->phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta),
			 0., (jet->phi()>0 ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta) );
	 container->AddElement( marker );
	 fw::addRhoZEnergyProjection( container, r_ecal, z_ecal, min_theta-0.003, max_theta+0.003, 
				       jet->phi(), iItem->defaultDisplayProperties().color() );
      }
      tList->AddElement(container);
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
   std::vector<CaloTowerRef> towers = jet.getConstituents();
   std::vector<double> phis;
   for ( std::vector<CaloTowerRef>::const_iterator tower = towers.begin(); 
	 tower != towers.end(); ++tower )
     phis.push_back( (*tower)->phi() );
   
   return fw::getPhiRange( phis, jet.phi() );
}
