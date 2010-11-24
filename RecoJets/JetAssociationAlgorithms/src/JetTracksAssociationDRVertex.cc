// Associate jets with tracks by simple "dR" criteria
// Fedor Ratnikov (UMd), Aug. 28, 2007
// $Id: JetTracksAssociationDRVertex.cc,v 1.4.2.1 2009/02/23 12:59:13 bainbrid Exp $

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRVertex.h"
#include "DataFormats/Math/interface/deltaR.h"

// -----------------------------------------------------------------------------
//
JetTracksAssociationDRVertex::JetTracksAssociationDRVertex( double fDr ) 
  : JetTracksAssociationDR(fDr),
    propagatedTracks_()
{;}

// -----------------------------------------------------------------------------
//
JetTracksAssociationDRVertex::~JetTracksAssociationDRVertex() 
{;}

// -----------------------------------------------------------------------------
//
void JetTracksAssociationDRVertex::produce( Association* fAssociation, 
					    const Jets& fJets,
					    const Tracks& fTracks,
					    const TrackQuality& fQuality ) 
{
  JetRefs jets;
  createJetRefs( jets, fJets );
  TrackRefs tracks;
  createTrackRefs( tracks, fTracks, fQuality );
  produce( fAssociation, jets, tracks );
}

// -----------------------------------------------------------------------------
//
void JetTracksAssociationDRVertex::produce( Association* fAssociation, 
					    const JetRefs& fJets,
					    const TrackRefs& fTracks ) 
{
  //clear();
  propagateTracks( fTracks ); 
  associateTracksToJets( fAssociation, fJets, fTracks ); 
}

// -----------------------------------------------------------------------------
//
void JetTracksAssociationDRVertex::associateTracksToJet( reco::TrackRefVector& associated,
							 const reco::Jet& fJet,
							 const TrackRefs& fTracks ) 
{
  associated.clear();
  std::vector<math::RhoEtaPhiVector>::const_iterator ii = propagatedTracks_.begin();
  std::vector<math::RhoEtaPhiVector>::const_iterator jj = propagatedTracks_.end();
  for ( ; ii != jj; ++ii ) {
    uint32_t index = ii - propagatedTracks_.begin();
    double dR2 = deltaR2( fJet.eta(), fJet.phi(), ii->eta(), ii->phi() );
    if ( dR2 < mDeltaR2Threshold ) { associated.push_back( fTracks[index] ); }
  }
}

// -----------------------------------------------------------------------------
//
void JetTracksAssociationDRVertex::propagateTracks( const TrackRefs& fTracks ) 
{
  propagatedTracks_.clear();
  propagatedTracks_.reserve( fTracks.size() );
  TrackRefs::const_iterator ii = fTracks.begin();
  TrackRefs::const_iterator jj = fTracks.end();
  for ( ; ii != jj; ++ii ) {
    propagatedTracks_.push_back( math::RhoEtaPhiVector( (**ii).p(), (**ii).eta(), (**ii).phi() ) ); 
  }
}  


