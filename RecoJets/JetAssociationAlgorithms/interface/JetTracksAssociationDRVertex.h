// \class JetTracksAssociationDRVertex
// Associate jets with tracks by simple "delta R" criteria
// Fedor Ratnikov (UMd), Aug. 28, 2007
// $Id: JetTracksAssociationDRVertex.h,v 1.4.2.1 2009/02/23 12:59:13 bainbrid Exp $

#ifndef RecoJets_JetAssociationAlgorithms_JetTracksAssociationDRVertex_h
#define RecoJets_JetAssociationAlgorithms_JetTracksAssociationDRVertex_h

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDR.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include <vector>

class JetTracksAssociationDRVertex : public JetTracksAssociationDR {

 public:

  /// Constructor taking dR threshold as argument
  explicit JetTracksAssociationDRVertex( double dr_threshold );
  
  /// Destructor
  ~JetTracksAssociationDRVertex();

  /// Associates tracks to jets (using Handles as input)
  void produce( Association*,
		const Jets&,
		const Tracks&,
		const TrackQuality& );
  
  /// Associates tracks to jets
  void produce( Association*,
		const JetRefs&,
		const TrackRefs& );
  
  // Associates tracks to the given jet
  void associateTracksToJet( reco::TrackRefVector&,
			     const reco::Jet&,
			     const TrackRefs& );
  
  // Calculates track impact points at calorimeter face
  void propagateTracks( const TrackRefs& );
  
 private:
  
  std::vector<math::RhoEtaPhiVector> propagatedTracks_;
  
};

#endif // RecoJets_JetAssociationAlgorithms_JetTracksAssociationDRVertex_h
