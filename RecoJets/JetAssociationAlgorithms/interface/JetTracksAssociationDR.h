// \class JetTracksAssociationDR
// Associate jets with tracks by simple "delta R" criteria
// Fedor Ratnikov (UMd)
// $Id: JetTracksAssociationDR.h,v 1.2 2010/03/16 21:48:47 srappocc Exp $

#ifndef RecoJets_JetAssociationAlgorithms_JetTracksAssociationDR_h
#define RecoJets_JetAssociationAlgorithms_JetTracksAssociationDR_h

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class MagneticField;
class Propagator;

class JetTracksAssociationDR {
  
 public:

  // ---------- Con(de)structors ----------
  
  /// Constructor taking dR threshold as argument
  explicit JetTracksAssociationDR( double dr_threshold );
  
  /// Destructor
  virtual ~JetTracksAssociationDR();
  
  // ---------- Typedefs ----------
  
  /// Container for jet-track associations
  typedef reco::JetTracksAssociation::Container Association;
  
  /// Handle to jet collection
  typedef edm::Handle< edm::View<reco::Jet> > Jets;
  
  /// Handle to track collection
  typedef edm::Handle< reco::TrackCollection > Tracks;
  
  // Jet reference
  typedef edm::RefToBase<reco::Jet> JetRef;
  
  // Collection of jet references
  typedef std::vector<JetRef> JetRefs;
  
  // Collection of track references
  typedef std::vector<reco::TrackRef> TrackRefs;

  // Track Quality
  typedef reco::TrackBase::TrackQuality TrackQuality;

  // ---------- Public interface ----------
  
  // Associate tracks to jets
  void associateTracksToJets( Association*,
			      const JetRefs&,
			      const TrackRefs& );
  
  /// Associate tracks to the given jet
  virtual void associateTracksToJet( reco::TrackRefVector&,
				     const reco::Jet&,
				     const TrackRefs& ) = 0;

  // Takes Handle as input and creates collection of edm::Refs
  static void createJetRefs( JetRefs&, 
			     const Jets& );
  
  // Takes Handle as input and creates collection of edm::Refs
  static void createTrackRefs( TrackRefs&,
			       const Tracks&, 
			       const TrackQuality& );
  
 protected:
  
  /// Private default constructor
  JetTracksAssociationDR() {}
  
  /// Threshold used to associate tracks to jets
  double mDeltaR2Threshold;
  
};

#endif // RecoJets_JetAssociationAlgorithms_JetTracksAssociationDR_h
