// \class JetTracksAssociationExplicit
// Associate jets with tracks of PFJet type explicitly by those
// in the PFCandidate constituents. 

#ifndef JetTracksAssociationExplicit_h
#define JetTracksAssociationExplicit_h

#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

class JetTracksAssociationExplicit {
 public:
  JetTracksAssociationExplicit ();
  ~JetTracksAssociationExplicit () {}

  void produce (reco::JetTracksAssociation::Container* fAssociation, 
		const std::vector <edm::RefToBase<reco::Jet> >& fJets,
		const std::vector <reco::TrackRef>& fTracks) const;
};

#endif
