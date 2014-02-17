// Associate jets with tracks by simple "dR" criteria
// Fedor Ratnikov (UMd), Aug. 28, 2007
// $Id: JetTracksAssociationDR.cc,v 1.1 2009/03/30 15:06:33 bainbrid Exp $

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// -----------------------------------------------------------------------------
//
JetTracksAssociationDR::JetTracksAssociationDR( double fDr ) 
  : mDeltaR2Threshold(fDr*fDr)
{;}

// -----------------------------------------------------------------------------
//
JetTracksAssociationDR::~JetTracksAssociationDR() 
{;}

// -----------------------------------------------------------------------------
//
void JetTracksAssociationDR::associateTracksToJets( Association* fAssociation, 
						    const JetRefs& fJets,
						    const TrackRefs& fTracks ) 
{
  JetRefs::const_iterator ii = fJets.begin();
  JetRefs::const_iterator jj = fJets.end();
  for ( ; ii != jj; ++ii ) {
    reco::TrackRefVector associated;
    associateTracksToJet( associated, **ii, fTracks );
    reco::JetTracksAssociation::setValue( fAssociation, *ii, associated );
  }
}

// -----------------------------------------------------------------------------
//
void JetTracksAssociationDR::createJetRefs( JetRefs& output, 
					    const Jets& input ) {
  output.clear();
  output.reserve( input->size() );
  for ( unsigned ii = 0; ii < input->size(); ++ii ) { 
    output.push_back( input->refAt(ii) );
  }
}

// -----------------------------------------------------------------------------
//
void JetTracksAssociationDR::createTrackRefs( TrackRefs& output,
					      const Tracks& input,
					      const TrackQuality& quality ) {

  if ( quality == reco::TrackBase::undefQuality ) {
    edm::LogError("JetTracksAssociationDR")
      << " Unknown TrackQuality value: " 
      << static_cast<int>( quality )
      << ". See possible values in 'reco::TrackBase::TrackQuality'";
  }

  output.clear();
  output.reserve( input->size() );
  for ( unsigned ii = 0; ii < input->size(); ++ii ) { 
    if ( (*input)[ii].quality( quality ) ) { 
      output.push_back( reco::TrackRef( input, ii ) );
    }
  }

}
