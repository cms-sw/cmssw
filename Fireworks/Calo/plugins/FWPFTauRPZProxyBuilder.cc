// -*- C++ -*-
// $Id: FWPFTauRPZProxyBuilder.cc,v 1.4 2009/10/04 12:13:19 dmytro Exp $
//

// include files
#include "Fireworks/Core/interface/FWRPZ2DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "Fireworks/Core/src/CmsShowMain.h"
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

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/PatCandidates/interface/Jet.h"


class FWPFTauRPZProxyBuilder : public FWRPZ2DSimpleProxyBuilderTemplate<reco::PFTau>  {
   
public:
  FWPFTauRPZProxyBuilder();	
  virtual ~FWPFTauRPZProxyBuilder();
  
  static std::pair<double,double>        getiEtaRange( const reco::PFJet& jet );
  static std::pair<double,double>  getPhiRange( const reco::PFJet& jet);
  void buildTauRhoPhi(const FWEventItem* iItem,
			     const reco::PFTau* iData,
			     TEveElement& tList) const;
  
  void buildTauRhoZ(  const FWEventItem* iItem,
			     const reco::PFTau* iData,
			     TEveElement& tList) const;
  REGISTER_PROXYBUILDER_METHODS();
  
private:
  FWPFTauRPZProxyBuilder(const FWPFTauRPZProxyBuilder&); // stop default
  
  const FWPFTauRPZProxyBuilder& operator=(const FWPFTauRPZProxyBuilder&); // stop default
  
  void buildRhoPhi(const reco::PFTau& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
  void buildRhoZ(const reco::PFTau& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
};

//
// constructors and destructor
//
FWPFTauRPZProxyBuilder::FWPFTauRPZProxyBuilder()
{
}

FWPFTauRPZProxyBuilder::~FWPFTauRPZProxyBuilder()
{
}

void 
FWPFTauRPZProxyBuilder::buildRhoPhi(const reco::PFTau& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
  buildTauRhoPhi( item(), &iData, oItemHolder);
}

void 
FWPFTauRPZProxyBuilder::buildRhoZ(const reco::PFTau& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
  buildTauRhoZ( item(), &iData, oItemHolder);
}

void 
FWPFTauRPZProxyBuilder::buildTauRhoPhi(const FWEventItem* iItem,
				      const reco::PFTau* tau,
				      TEveElement& container) const
{
  const reco::PFTauTagInfo *tau_tag_info = (const reco::PFTauTagInfo*)(tau->pfTauTagInfoRef().get());
  const reco::PFJet *jet = (const reco::PFJet*)(tau_tag_info->pfjetRef().get());
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
FWPFTauRPZProxyBuilder::buildTauRhoZ(const FWEventItem* iItem,
				      const reco::PFTau* tau,
				      TEveElement& container) const
{
   const reco::PFTauTagInfo *tau_tag_info = (const reco::PFTauTagInfo*)(tau->pfTauTagInfoRef().get());
   const reco::PFJet *jet = (const reco::PFJet*)(tau_tag_info->pfjetRef().get());
   // NOTE:
   //      We derive eta bin size from xbins array used for LEGO assuming that all 82
   //      eta bins are accounted there.
   static const int nBins = sizeof(fw3dlego::xbins)/sizeof(*fw3dlego::xbins);
   assert (  nBins == 82+1 );
   static const std::vector<std::pair<double,double> > thetaBins = fireworks::thetaBins();

   const double z_ecal = 306; // ECAL endcap inner surface
   const double r_ecal = 126;
   const double transition_angle = atan(r_ecal/z_ecal);

   std::pair<double,double> iEtaRange = getiEtaRange( *jet );

   double max_theta = iEtaRange.first;
   double min_theta = iEtaRange.second;

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

std::pair<double,double>
FWPFTauRPZProxyBuilder::getiEtaRange( const reco::PFJet& jet )
{
   double min =  100;
   double max = -100;

  
   std::vector <const reco::Candidate*> Candidates;
   Candidates = jet.getJetConstituentsQuick();
   for ( std::vector<const reco::Candidate*>::const_iterator candidate = Candidates.begin();
	 candidate != Candidates.end(); ++candidate ){
     double itheta = (*candidate)->theta();
     if ( itheta > max ) max = itheta;
     if ( itheta < min ) min = itheta;
     
   }
   
   if ( min > max ) return std::pair<double,double>(0.,0.);
   return std::pair<double,double>(min,max);
}

std::pair<double,double>
FWPFTauRPZProxyBuilder::getPhiRange( const reco::PFJet& jet )
{
   
  std::vector <const reco::Candidate*> Candidates;
   std::vector<double> phis;
  
   Candidates = jet.getJetConstituentsQuick();
   for ( std::vector<const reco::Candidate*>::const_iterator candidate = Candidates.begin();
	 candidate != Candidates.end(); ++candidate ){
     phis.push_back( (*candidate)->phi() );
   }
   return fw::getPhiRange( phis, jet.phi() );
}


REGISTER_FWRPZDATAPROXYBUILDERBASE(FWPFTauRPZProxyBuilder,reco::PFTau,"PFTau");
