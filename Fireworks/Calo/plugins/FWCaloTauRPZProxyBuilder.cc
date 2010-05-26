// -*- C++ -*-
// $Id: FWCaloTauRPZProxyBuilder.cc,v 1.4 2009/10/04 12:13:19 dmytro Exp $
//

// include files
#include "Fireworks/Core/interface/FWRPZ2DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"

#include "TEveGeoNode.h"
#include "TGeoArb8.h"
#include "TEveManager.h"
#include "TGeoSphere.h"
#include "TGeoTube.h"
#include "TEvePointSet.h"
#include "TEveScalableStraightLineSet.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/fw3dlego_xbins.h"
#include "Fireworks/Calo/interface/thetaBins.h"

#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/PatCandidates/interface/Jet.h"


class FWCaloTauRPZProxyBuilder : public FWRPZ2DSimpleProxyBuilderTemplate<reco::CaloTau>  {
   
public:
  FWCaloTauRPZProxyBuilder();	
  virtual ~FWCaloTauRPZProxyBuilder();
  
  static std::pair<int,int>        getiEtaRange( const reco::CaloJet& jet );
  static std::pair<double,double>  getPhiRange( const reco::CaloJet& jet);
  void buildTauRhoPhi(const FWEventItem* iItem,
			     const reco::CaloTau* iData,
			     TEveElement& tList) const;
  
  void buildTauRhoZ(  const FWEventItem* iItem,
			     const reco::CaloTau* iData,
			     TEveElement& tList) const;
  REGISTER_PROXYBUILDER_METHODS();
  
private:
  FWCaloTauRPZProxyBuilder(const FWCaloTauRPZProxyBuilder&); // stop default
  
  const FWCaloTauRPZProxyBuilder& operator=(const FWCaloTauRPZProxyBuilder&); // stop default
  
  void buildRhoPhi(const reco::CaloTau& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
  void buildRhoZ(const reco::CaloTau& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
};

//
// constructors and destructor
//
FWCaloTauRPZProxyBuilder::FWCaloTauRPZProxyBuilder()
{
}

FWCaloTauRPZProxyBuilder::~FWCaloTauRPZProxyBuilder()
{
}

void 
FWCaloTauRPZProxyBuilder::buildRhoPhi(const reco::CaloTau& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
  buildTauRhoPhi( item(), &iData, oItemHolder);
}

void 
FWCaloTauRPZProxyBuilder::buildRhoZ(const reco::CaloTau& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
  buildTauRhoZ( item(), &iData, oItemHolder);
}

void 
FWCaloTauRPZProxyBuilder::buildTauRhoPhi(const FWEventItem* iItem,
				      const reco::CaloTau* tau,
				      TEveElement& container) const
{
  const reco::CaloTauTagInfo *tau_tag_info = (const reco::CaloTauTagInfo*)(tau->caloTauTagInfoRef().get());
  const reco::CaloJet *jet = (const reco::CaloJet*)(tau_tag_info->calojetRef().get());
  TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   const double r_ecal = 126;
   std::pair<double,double> phiRange = getPhiRange( *jet );
   double min_phi = phiRange.first-M_PI/36/2;
   double max_phi = phiRange.second+M_PI/36/2;
   double phi = jet->phi();
   if ( fabs(phiRange.first-phiRange.first)<1e-3 ) {
     min_phi = phi-M_PI/36/2;
     max_phi = phi+M_PI/36/2;
   }
   
   double size = jet->et();
   TGeoBBox *sc_box = new TGeoTubeSeg(r_ecal - 1, r_ecal + 1, 1, min_phi * 180 / M_PI, max_phi * 180 / M_PI);
   TEveGeoShape *element = fw::getShape( "spread", sc_box, iItem->defaultDisplayProperties().color() );
   element->SetPickable(kTRUE);
   container.AddElement(element);
   
   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
   marker->SetLineWidth(4);
   marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
   
   marker->SetScaleCenter( r_ecal*cos(phi), r_ecal*sin(phi), 0 );
   marker->AddLine( r_ecal*cos(phi), r_ecal*sin(phi), 0, (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
   container.AddElement(marker);
   
   const reco::TrackRef lead_track = tau->leadTrack();
   reco::TrackRefVector::iterator tracks_end = tau->signalTracks().end(); 
   for (reco::TrackRefVector::iterator i = tau->signalTracks().begin(); i != tracks_end; ++i ){
     
     TEveTrack* track(0);
     if ( i->isAvailable() )
       {
	 track = fireworks::prepareTrack(**i,
					 context().getTrackPropagator(),
					 item()->defaultDisplayProperties().color() );
       }
     track->MakeTrack();
     if(track)container.AddElement(track);
     
   }
   
}

void 
FWCaloTauRPZProxyBuilder::buildTauRhoZ(const FWEventItem* iItem,
				      const reco::CaloTau* tau,
				      TEveElement& container) const
{
   const reco::CaloTauTagInfo *tau_tag_info = (const reco::CaloTauTagInfo*)(tau->caloTauTagInfoRef().get());
   const reco::CaloJet *jet = (const reco::CaloJet*)(tau_tag_info->calojetRef().get());
   // NOTE:
   //      We derive eta bin size from xbins array used for LEGO assuming that all 82
   //      eta bins are accounted there.
   static const int nBins = sizeof(fw3dlego::xbins)/sizeof(*fw3dlego::xbins);
   assert (  nBins == 82+1 );
   static const std::vector<std::pair<double,double> > thetaBins = fireworks::thetaBins();

   const double z_ecal = 306; // ECAL endcap inner surface
   const double r_ecal = 126;
   const double transition_angle = atan(r_ecal/z_ecal);

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

   double size = jet->et();

   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
   marker->SetLineWidth(4);
   marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
   marker->SetScaleCenter( 0., (jet->phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta) );
   marker->AddLine(0., (jet->phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta),
                   0., (jet->phi()>0 ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta) );
   container.AddElement( marker );
   fw::addRhoZEnergyProjection( &container, r_ecal, z_ecal, min_theta-0.003, max_theta+0.003,
                                jet->phi(), iItem->defaultDisplayProperties().color() );

   
   const reco::TrackRef lead_track = tau->leadTrack();
   reco::TrackRefVector::iterator tracks_end = tau->signalTracks().end(); 
   for (reco::TrackRefVector::iterator i = tau->signalTracks().begin(); i != tracks_end; ++i ){
     
     TEveTrack* track(0);
     if ( i->isAvailable() )
       {
	 track = fireworks::prepareTrack(**i,
					 context().getTrackPropagator(),
					 item()->defaultDisplayProperties().color() );
       }
     track->MakeTrack();
     if(track)container.AddElement(track);
     
   }

}

std::pair<int,int>
FWCaloTauRPZProxyBuilder::getiEtaRange( const reco::CaloJet& jet )
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
FWCaloTauRPZProxyBuilder::getPhiRange( const reco::CaloJet& jet )
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


REGISTER_FWRPZDATAPROXYBUILDERBASE(FWCaloTauRPZProxyBuilder,reco::CaloTau,"CaloTau");
