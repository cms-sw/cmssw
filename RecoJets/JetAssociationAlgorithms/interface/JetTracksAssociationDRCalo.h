// \class JetTracksAssociationDRCalo
// Associate jets with tracks by simple "delta R" criteria
// Fedor Ratnikov (UMd), Aug. 28, 2007
// $Id: JetTracksAssociationDRCalo.h,v 1.2 2007/09/11 23:59:19 fedor Exp $

#ifndef JetTracksAssociationDRCalo_h
#define JetTracksAssociationDRCalo_h

#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

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
 private:
  /// fidutial dR between track in the vertex and jet's reference direction
  double mDeltaR2Threshold;
};

#endif
