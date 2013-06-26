// Associate jets with tracks by simple "dR" criteria
// Fedor Ratnikov (UMd), Aug. 28, 2007
// $Id: JetTracksAssociationXtrpCalo.cc,v 1.4 2011/02/26 05:59:34 dlange Exp $

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationXtrpCalo.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

// -----------------------------------------------------------------------------
//
JetTracksAssociationXtrpCalo::JetTracksAssociationXtrpCalo() 
{}

// -----------------------------------------------------------------------------
//
JetTracksAssociationXtrpCalo::~JetTracksAssociationXtrpCalo() 
{}


// -----------------------------------------------------------------------------
//

void JetTracksAssociationXtrpCalo::produce( Association* fAssociation,
					    JetRefs const & fJets,
					    std::vector<reco::TrackExtrapolation> const & fExtrapolations,
					    CaloGeometry const & fGeo,
					    double dR )
{
  for ( JetRefs::const_iterator jetsBegin = fJets.begin(),
	  jetsEnd = fJets.end(), 
	  ijet = jetsBegin;
	ijet != jetsEnd; ++ijet ) {
    reco::TrackRefVector associated;
    associateInputTracksToJet( associated, **ijet, fExtrapolations, dR );
    reco::JetTracksAssociation::setValue( fAssociation, *ijet, associated );
  }
  
  
}



void JetTracksAssociationXtrpCalo::associateInputTracksToJet( reco::TrackRefVector& associated,
							      const reco::Jet& fJet,
							      std::vector<reco::TrackExtrapolation> const & fExtrapolations,
							      double dR ) 
{
  reco::CaloJet const * pCaloJet = dynamic_cast<reco::CaloJet const *>(&fJet);
  if ( pCaloJet == 0 ) {
    throw cms::Exception("InvalidInput") << "Expecting calo jets only in JetTracksAssociationXtrpCalo";
  }
  // Loop over CaloTowers
  double jetPhi = pCaloJet->phi();
  double jetEta = pCaloJet->detectorP4().eta();

  // now cache the mapping of (det ID --> track)

//  std::cout<<" New jet "<<jetEta<<" "<<jetPhi<<" Jet ET "<<pCaloJet->et()<<std::endl;

  for ( std::vector<reco::TrackExtrapolation>::const_iterator xtrpBegin = fExtrapolations.begin(),
	  xtrpEnd = fExtrapolations.end(), ixtrp = xtrpBegin;
	ixtrp != xtrpEnd; ++ixtrp ) {
	
    if ( ixtrp->positions().size()==0 ) continue;
    reco::TrackBase::Point const & point = ixtrp->positions().at(0);
    
    
    double dr = reco::deltaR<double>( jetEta, jetPhi, point.eta(), point.phi() );
    if ( dr < dR ) {

//    std::cout<<" JetTracksAssociationXtrpCalo::associateInputTracksToJet:: initial track "<<ixtrp->track()->pt()<<" "<<ixtrp->track()->eta()<<
//    " "<<ixtrp->track()->phi()<< " Extrapolated position "<<point.eta()<<" "<<point.phi()<<" Valid? "<<ixtrp->isValid().at(0)<<" Jet eta, phi "<<jetEta<<" "<<jetPhi<<" Jet ET "<<pCaloJet->et()<<
//    " dr "<<dr<<" dR "<<dR<<std::endl;

      reco::TrackRef matchedTrack = ixtrp->track(); 
      associated.push_back( matchedTrack );      
    }
  }


}
