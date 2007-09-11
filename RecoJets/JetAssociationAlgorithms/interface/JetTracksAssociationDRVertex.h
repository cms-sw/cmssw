// \class JetTracksAssociationDRVertex
// Associate jets with tracks by simple "delta R" criteria
// Fedor Ratnikov (UMd), Aug. 28, 2007
// $Id: JetTracksAssociationDRVertex.h,v 1.1 2007/08/29 17:53:13 fedor Exp $

#ifndef JetTracksAssociationDRVertex_h
#define JetTracksAssociationDRVertex_h

#include "DataFormats/JetReco/interface/JetToTracksAssociation.h"

class JetTracksAssociationDRVertex {
 public:
  JetTracksAssociationDRVertex (double fDr);
  ~JetTracksAssociationDRVertex () {}

  void produce (reco::JetToTracksAssociation::Container* fAssociation, 
		const std::vector <edm::RefToBase<reco::Jet> >& fJets,
		const std::vector <reco::TrackRef>& fTracks) const;
 private:
  /// fidutial dR between track in the vertex and jet's reference direction
  double mDeltaR2Threshold;
};

#endif
