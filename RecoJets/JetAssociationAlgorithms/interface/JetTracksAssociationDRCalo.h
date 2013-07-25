// \class JetTracksAssociationDRCalo
// Associate jets with tracks by simple "delta R" criteria
// Fedor Ratnikov (UMd), Aug. 28, 2007
// $Id: JetTracksAssociationDRCalo.h,v 1.6 2010/03/18 12:17:58 bainbrid Exp $

#ifndef JetTracksAssociationDRCalo_h
#define JetTracksAssociationDRCalo_h

#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/Math/interface/Point3D.h"

class MagneticField;
class Propagator;

class JetTracksAssociationDRCalo {
 public:
  JetTracksAssociationDRCalo (double fDr);
  ~JetTracksAssociationDRCalo () {}

  void produce (reco::JetTracksAssociation::Container* fAssociation, 
		const std::vector <edm::RefToBase<reco::Jet> >& fJets,
		const std::vector <reco::TrackRef>& fTracks,
		const MagneticField& fField,
		const Propagator& fPropagator) const;

  /// propagating the track to the Calorimeter
  static math::XYZPoint propagateTrackToCalorimeter (const reco::Track& fTrack,
						     const MagneticField& fField,
						     const Propagator& fPropagator);
 private:
  /// fidutial dR between track in the vertex and jet's reference direction
  double mDeltaR2Threshold;
};

#endif
