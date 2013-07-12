// \class JetTracksAssociationDRVertex
// Associate jets with tracks by simple "delta R" criteria.
// This is different from the "JetTracksAssociatorAtVertex" because this
// class assigns a vertex to the jet/track association. 
// Fedor Ratnikov (UMd), Aug. 28, 2007
// $Id: JetTracksAssociationDRVertex.h,v 1.5 2010/03/18 12:17:58 bainbrid Exp $

#ifndef JetTracksAssociationDRVertexAssigned_h
#define JetTracksAssociationDRVertexAssigned_h

#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
class JetTracksAssociationDRVertexAssigned {
 public:
  JetTracksAssociationDRVertexAssigned (double fDr);
  ~JetTracksAssociationDRVertexAssigned () {}

  void produce (reco::JetTracksAssociation::Container* fAssociation, 
		const std::vector <edm::RefToBase<reco::Jet> >& fJets,
		const std::vector <reco::TrackRef>& fTracks,
                const reco::VertexCollection& vertices) const;
 private:
  /// fidutial dR between track in the vertex and jet's reference direction
  double mDeltaR2Threshold;
};

#endif
