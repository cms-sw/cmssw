#ifndef RecoBTag_SoftLepton_SoftLeptonAlgorithm_h
#define RecoBTag_SoftLepton_SoftLeptonAlgorithm_h

// -*- C++ -*-
//
// Package:    SoftLepton
// Class:      SoftLeptonAlgorithm
// 
/**\class SoftLepton SoftLeptonAlgorithm.h RecoBTag/SoftLepton/interface/SoftLeptonAlgorithm.h

 Description: Concrete implementation of soft lepton b tagging algorithm

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  fwyzard
//         Created:  Wed Oct 18 18:02:07 CEST 2006
// $Id: SoftLeptonAlgorithm.h,v 1.8 2007/05/11 11:47:14 fwyzard Exp $
//

#include <utility>

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

class TransientTrackBuilder;
class JetTagComputer;
class TVector3;

class SoftLeptonAlgorithm {
public:

  SoftLeptonAlgorithm( void ) : 
    m_transientTrackBuilder( NULL ),
    m_concreteTagger( NULL ),
    m_refineJetAxis( reco::SoftLeptonProperties::AXIS_CHARGED )
  {
  }

  ~SoftLeptonAlgorithm( void ) {
  }

  void setTransientTrackBuilder( const TransientTrackBuilder * builder ) { 
    m_transientTrackBuilder = builder; 
  }

  void setConcreteTagger( const JetTagComputer * tagger ) {
    m_concreteTagger = tagger;
  }

  void refineJetAxis( unsigned int axis ) {
    m_refineJetAxis = axis;
  }
  
  void setDeltaRCut( double cut ) {
    m_deltaRCut = cut;
  }

  // generic interface, using a TrackRefVector for lepton tracks
  std::pair < reco::JetTag, reco::SoftLeptonTagInfo > 
  tag( 
      const reco::JetTracksAssociationRef & jetTracks, 
      const reco::Vertex                  & primaryVertex,
      const reco::TrackRefVector          & leptons
  );
    
private:
  // service used to make transient tracks from tracks
  const TransientTrackBuilder * m_transientTrackBuilder;

  // service used to compute the discriminator from the tagging variables
  const JetTagComputer * m_concreteTagger;

  // algorithm configuration
  unsigned int m_refineJetAxis;
  double       m_deltaRCut;
  
public:
  static GlobalVector refineJetAxis (
      const edm::RefToBase<reco::Jet> & jet,
      const reco::TrackRefVector      & tracks, 
      const reco::TrackRef            & excluded = reco::TrackRef()
  );

  static double relativeEta(
      const TVector3 & a, 
      const TVector3 & b
  );
  
};

#endif // RecoBTag_SoftLepton_SoftLeptonAlgorithm_h
