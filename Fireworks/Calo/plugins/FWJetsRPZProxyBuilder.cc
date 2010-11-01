// -*- C++ -*-
// $Id: FWJetsRPZProxyBuilder.cc,v 1.6 2009/08/29 21:01:15 dmytro Exp $
//

// include files
#include "Fireworks/Core/interface/FWRPZ2DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "TEveGeoNode.h"
#include "TGeoArb8.h"
#include "TEveManager.h"
#include "TGeoSphere.h"
#include "TGeoTube.h"
#include "TH1F.h"
#include "TColor.h"
#include "TROOT.h"
#include "TEvePointSet.h"
#include "TEveScalableStraightLineSet.h"
#include "TEveCompound.h"
#include <boost/shared_ptr.hpp>
#include <boost/mem_fn.hpp>

// user include files
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Calo/interface/thetaBins.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "Fireworks/Core/interface/FWRhoPhiZView.h"
#include "Fireworks/Calo/interface/FW3DEveJet.h"

#include "Fireworks/Core/interface/fw3dlego_xbins.h"

class FWJetsRPZProxyBuilder : public FWRPZ2DSimpleProxyBuilderTemplate<reco::Jet>  {

public:
   FWJetsRPZProxyBuilder(){
   }
   virtual ~FWJetsRPZProxyBuilder(){
   }

   static std::pair<int,int>        getiEtaRange( const reco::Jet& jet );
   static std::pair<double,double>  getPhiRangeByConstituents( const reco::Jet& jet );
   static std::pair<double,double>  getPhiRangeByMoments( const reco::Jet& jet );
   static double getTheta( double eta ) {
      return 2*atan(exp(-eta));
   }

   static void buildJetRhoPhi(const FWEventItem* iItem,
                              const reco::Jet* jet,
                              TEveElement& tList);

   static void buildJetRhoZ(  const FWEventItem* iItem,
                              const reco::Jet* jet,
                              TEveElement& tList);

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWJetsRPZProxyBuilder(const FWJetsRPZProxyBuilder&); // stop default

   const FWJetsRPZProxyBuilder& operator=(const FWJetsRPZProxyBuilder&); // stop default

   void buildRhoPhi(const reco::Jet& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
   void buildRhoZ(const reco::Jet& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
};

void
FWJetsRPZProxyBuilder::buildRhoPhi(const reco::Jet& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   buildJetRhoPhi( item(), &iData, oItemHolder);
}

void
FWJetsRPZProxyBuilder::buildRhoZ(const reco::Jet& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   buildJetRhoZ( item(), &iData, oItemHolder);
}

void
FWJetsRPZProxyBuilder::buildJetRhoPhi(const FWEventItem* iItem,
                                      const reco::Jet* jet,
                                      TEveElement& container)
{
   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   const double r_ecal = 126;
   // std::pair<double,double> phiRange = getPhiRangeByConstituents( *jet );
   std::pair<double,double> phiRange = getPhiRangeByMoments( *jet );
   
   double min_phi = phiRange.first-M_PI/36/2;
   double max_phi = phiRange.second+M_PI/36/2;
   double phi = jet->phi();
   if ( fabs(phiRange.first-phiRange.second)<1e-3 ) {
     min_phi = phi-M_PI/36/2;
     max_phi = phi+M_PI/36/2;
   }

   double size = jet->et();

   // TGeoBBox *sc_box = new TGeoTubeSeg(r_ecal - 1, r_ecal + 1, 1, min_phi * 180 / M_PI, max_phi * 180 / M_PI);
   // TEveGeoShape *element = fw::getShape( "spread", sc_box, iItem->defaultDisplayProperties().color() );
   // element->SetPickable(kTRUE);
   // container.AddElement(element);
   
   // TGeoSphere* shape = new TGeoSphere(double(0),r_ecal, 89.99, 90.01, min_phi * 180 / M_PI, max_phi * 180 / M_PI );
   // shape->SetNumberOfDivisions(1);

   Double_t points[16];
   points[0] = jet->vertex().x();
   points[1] = jet->vertex().y();
   points[2] = r_ecal*cos(min_phi);
   points[3] = r_ecal*sin(min_phi);
   points[4] = r_ecal*cos(max_phi);
   points[5] = r_ecal*sin(max_phi);
   points[6] = jet->vertex().x();
   points[7] = jet->vertex().y();
   for( int i = 0; i<8; ++i ){
     points[i+8] = points[i];
   }

   TEveGeoShape *element = fw::getShape( "cone", 
					 new TGeoArb8(0,points),
					 iItem->defaultDisplayProperties().color() );
   element->SetMainTransparency(90);
   element->SetPickable(kTRUE);
   container.AddElement(element);

   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
   marker->SetLineWidth(4);
   marker->SetLineColor(  iItem->defaultDisplayProperties().color() );

   marker->SetScaleCenter( r_ecal*cos(phi), r_ecal*sin(phi), 0 );
   marker->AddLine( r_ecal*cos(phi), r_ecal*sin(phi), 0, (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
   container.AddElement(marker);

   // FW3DEveJet* cone = new FW3DEveJet(*jet,"jetcone","jetcone");
   // cone->SetPickable(kTRUE);
   // cone->SetMainColor( iItem->defaultDisplayProperties().color() );
   // cone->SetMainTransparency(75);
   // container.AddElement(cone);
}


void
FWJetsRPZProxyBuilder::buildJetRhoZ(const FWEventItem* iItem,
                                    const reco::Jet* jet,
                                    TEveElement& container)
{
   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   // NOTE:
   //      We derive eta bin size from xbins array used for LEGO assuming that all 82
   //      eta bins are accounted there.
   static const int nBins = sizeof(fw3dlego::xbins)/sizeof(*fw3dlego::xbins);
   assert (  nBins == 82+1 );
   static const std::vector<std::pair<double,double> > thetaBins = fireworks::thetaBins();

   const double z_ecal = 306; // ECAL endcap inner surface
   const double r_ecal = 126;
   const double transition_angle = atan(r_ecal/z_ecal);

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

   double size = jet->et();
   double etaSize = sqrt(jet->etaetaMoment());

   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
   marker->SetLineWidth(4);
   marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
   marker->SetScaleCenter( 0., (jet->phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta) );
   marker->AddLine(0., (jet->phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta),
                   0., (jet->phi()>0 ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta) );
   container.AddElement( marker );

   double min_theta = getTheta(jet->eta()+etaSize);
   double max_theta = getTheta(jet->eta()-etaSize);
   Double_t points[16];
   points[0] = jet->vertex().z();
   points[1] = jet->phi()>0 ? jet->vertex().rho() : -jet->vertex().rho();
      
   if ( max_theta > M_PI - transition_angle )
     {
       points[6] = -z_ecal;
       points[7] = jet->phi()>0 ? z_ecal*fabs(tan(max_theta)) : -z_ecal*fabs(tan(max_theta));
     } else 
     if ( max_theta < transition_angle ) {
       points[6] = z_ecal;
       points[7] = jet->phi()>0 ? z_ecal*fabs(tan(max_theta)) : -z_ecal*fabs(tan(max_theta));
     } else {
       points[6] = r_ecal/tan(max_theta);
       points[7] = jet->phi()>0 ? r_ecal : -r_ecal;
     }
      
   if ( min_theta > M_PI - transition_angle )
     {
       points[2] = -z_ecal;
       points[3] = jet->phi()>0 ? z_ecal*fabs(tan(min_theta)) : -z_ecal*fabs(tan(min_theta));
     } else 
     if ( min_theta < transition_angle ) {
       points[2] = z_ecal;
       points[3] = jet->phi()>0 ? z_ecal*fabs(tan(min_theta)) : -z_ecal*fabs(tan(min_theta));
     } else {
       points[2] = r_ecal/tan(min_theta);
       points[3] = jet->phi()>0 ? r_ecal : -r_ecal;
     }
      
   if ( min_theta < M_PI - transition_angle && max_theta > M_PI - transition_angle )
     {
       points[4] = -z_ecal;
       points[5] = jet->phi()>0 ? r_ecal : -r_ecal;
     } else
     if ( min_theta < transition_angle && max_theta > transition_angle ) {
       points[4] = z_ecal;
       points[5] = jet->phi()>0 ? r_ecal : -r_ecal;
     } else {
       points[4] = points[2];
       points[5] = points[3];
     }
      
   for( int i = 0; i<8; ++i ){
     points[i+8] = points[i];
   }
   TEveGeoShape *element = fw::getShape( "cone2", 
					 new TGeoArb8(0,points),
					 iItem->defaultDisplayProperties().color() );
   element->RefMainTrans().RotateLF(1,3,M_PI/2 );
   element->SetMainTransparency(90);
   element->SetPickable(kTRUE);
   container.AddElement(element);

   // std::pair<int,int> iEtaRange = getiEtaRange( *jet );
   // double max_theta = thetaBins[iEtaRange.first].first;
   // double min_theta = thetaBins[iEtaRange.second].second;;
   // fw::addRhoZEnergyProjection( &container, r_ecal, z_ecal, min_theta-0.003, max_theta+0.003,
   //                               jet->phi(), iItem->defaultDisplayProperties().color() );

   // FW3DEveJet* cone = new FW3DEveJet(*jet,"jetcone","jetcone");
   // cone->SetPickable(kTRUE);
   // cone->SetMainColor( iItem->defaultDisplayProperties().color() );
   // cone->SetMainTransparency(75);
   // container.AddElement(cone);
}

std::pair<int,int>
FWJetsRPZProxyBuilder::getiEtaRange( const reco::Jet& jet )
{
   int min =  100;
   int max = -100;

   std::vector<CaloTowerPtr> towers;
   try {
     if ( const reco::CaloJet* calojet = dynamic_cast<const reco::CaloJet*>(&jet) )
       towers = calojet->getCaloConstituents();
     else {
       if ( const pat::Jet* patjet = dynamic_cast<const pat::Jet*>(&jet) ){
	 if ( patjet->isCaloJet() )
	   towers = patjet->getCaloConstituents();
       }
     }
   }
   catch (...) {
     std::cout << "Failed to get calo jet constituents." << std::endl;
   }


   for ( std::vector<CaloTowerPtr>::const_iterator tower = towers.begin();
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

std::pair<double,double>
FWJetsRPZProxyBuilder::getPhiRangeByConstituents( const reco::Jet& jet )
{
   std::vector<CaloTowerPtr> towers;
   try {
     if ( const reco::CaloJet* calojet = dynamic_cast<const reco::CaloJet*>(&jet) )
       towers = calojet->getCaloConstituents();
     else {
       if ( const pat::Jet* patjet = dynamic_cast<const pat::Jet*>(&jet) ){
	 if ( patjet->isCaloJet() )
	   towers = patjet->getCaloConstituents();
       }
     }
   }
   catch (...) {
     std::cout << "Failed to get calo jet constituents." << std::endl;
   }
   std::vector<double> phis;
   for ( std::vector<CaloTowerPtr>::const_iterator tower = towers.begin();
         tower != towers.end(); ++tower )
      phis.push_back( (*tower)->phi() );

   return fw::getPhiRange( phis, jet.phi() );
}
std::pair<double,double>
FWJetsRPZProxyBuilder::getPhiRangeByMoments( const reco::Jet& jet )
{
   std::vector<double> phis;
   double phiSize = sqrt(jet.phiphiMoment());
   phis.push_back(jet.phi()+phiSize);
   phis.push_back(jet.phi()-phiSize);
   
   return fw::getPhiRange( phis, jet.phi() );
}

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWJetsRPZProxyBuilder,reco::Jet,"Jets");
