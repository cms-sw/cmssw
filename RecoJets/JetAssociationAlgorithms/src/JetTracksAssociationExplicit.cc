// Associate jets with tracks by simple "dR" criteria
// Fedor Ratnikov (UMd), Aug. 28, 2007

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationExplicit.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/TrackReco/interface/Track.h"

JetTracksAssociationExplicit::JetTracksAssociationExplicit () 
{}

void JetTracksAssociationExplicit::produce (reco::JetTracksAssociation::Container* fAssociation, 
					    const std::vector <edm::RefToBase<reco::Jet> >& fJets,
					    const std::vector <reco::TrackRef>& fTracks) const 
{
  for (unsigned j = 0; j < fJets.size(); ++j) { 
    reco::PFJet const * pfJet = dynamic_cast<reco::PFJet const *>( &* (fJets[j]) ) ;
    if ( pfJet != nullptr ) {
      reco::TrackRefVector assoTracks = pfJet->getTrackRefs();
      reco::JetTracksAssociation::setValue (fAssociation, fJets[j], assoTracks);
    } else {
      throw cms::Exception("InvalidConfiguration") 
	<< "From JetTracksAssociationExplicit::produce: Only PFJets are currently supported for this module" << std::endl;
    }
  }
}
