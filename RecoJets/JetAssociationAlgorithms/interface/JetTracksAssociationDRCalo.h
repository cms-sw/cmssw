// \class JetTracksAssociationDRCalo
// Associate jets with tracks by simple "delta R" criteria
// Fedor Ratnikov (UMd), Aug. 28, 2007
// $Id: JetTracksAssociationDRCalo.h,v 1.4.2.1 2009/02/23 12:59:13 bainbrid Exp $

#ifndef RecoJets_JetAssociationAlgorithms_JetTracksAssociationDRCalo_h
#define RecoJets_JetAssociationAlgorithms_JetTracksAssociationDRCalo_h

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDR.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/Point3D.h"
#include <vector>

class MagneticField;
class Propagator;

class JetTracksAssociationDRCalo : public JetTracksAssociationDR {
  
 public:
  
  /// Constructor taking dR threshold as argument
  explicit JetTracksAssociationDRCalo( double dr_threshold );
  
  /// Destructor
  ~JetTracksAssociationDRCalo();

  /// Associates tracks to jets (using Handles as input)
  void produce( Association*,
		const Jets&,
		const Tracks&,
		const TrackQuality&,
		const MagneticField&,
		const Propagator& );

  /// Associates tracks to jets
  void produce( Association*,
		const JetRefs&,
		const TrackRefs&,
		const MagneticField&,
		const Propagator& );
  
  // Associates tracks to the given jet
  void associateTracksToJet( reco::TrackRefVector&,
			     const reco::Jet&,
			     const TrackRefs& );
  
  // Calculates track impact points at calorimeter face
  void propagateTracks( const TrackRefs&,
			const MagneticField&,
			const Propagator& );
  
  /// Propagates track to calorimeter face
  static math::XYZPoint propagateTrackToCalorimeter( const reco::Track&,
						     const MagneticField&,
						     const Propagator& );
  
 private:

  /// Private default constructor
  JetTracksAssociationDRCalo();

  /// Propagates track to calorimeter face
  static GlobalPoint propagateTrackToCalo( const reco::Track&,
					   const MagneticField&,
					   const Propagator& );
  
  /// Definition of track impact point 
  struct ImpactPoint {
    unsigned index;
    double eta;
    double phi;
  };
  
  /// Impact points of tracks at calorimeter face
  std::vector<ImpactPoint> propagatedTracks_;
  
};

#endif // RecoJets_JetAssociationAlgorithms_JetTracksAssociationDRCalo_h
