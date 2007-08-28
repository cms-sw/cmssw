#ifndef RecoBTag_SoftLepton_SoftLeptonAlgorithm_h
#define RecoBTag_SoftLepton_SoftLeptonAlgorithm_h

#include <utility>

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/Jet.h"

class TransientTrackBuilder;
class JetTagComputer;

class SoftLeptonAlgorithm {
public:

  SoftLeptonAlgorithm( void ) : 
    m_transientTrackBuilder( NULL ),
    m_refineJetAxis( reco::SoftLeptonProperties::AXIS_CALORIMETRIC ),
    m_deltaRCut( 0.7 ),
    m_chi2Cut( 0.0 ),
    m_usePrimaryVertex( true )
  {
  }

  SoftLeptonAlgorithm( const edm::ParameterSet & iConfig );

  ~SoftLeptonAlgorithm( void ) {
  }

  void setTransientTrackBuilder( const TransientTrackBuilder * builder ) { 
    m_transientTrackBuilder = builder; 
  }

  // generic interface, using a TrackRefVector for lepton tracks
  reco::SoftLeptonTagInfo tag (
      const edm::RefToBase<reco::Jet> & jet,
      const reco::TrackRefVector      & tracks,
      const reco::TrackRefVector      & leptons,
      const reco::Vertex              & primaryVertex
  ) const;
    
protected:
  
  GlobalVector refineJetAxis (
      const edm::RefToBase<reco::Jet> & jet,
      const reco::TrackRefVector      & tracks, 
      const reco::TrackRef            & excluded = reco::TrackRef()
  ) const;

  static double relativeEta(
      const math::XYZVector& vector, 
      const math::XYZVector& axis
  );

  // service used to make transient tracks from tracks
  const TransientTrackBuilder * m_transientTrackBuilder;

  // algorithm configuration
  unsigned int m_refineJetAxis;
  double       m_deltaRCut;
  double       m_chi2Cut;
  bool         m_usePrimaryVertex; 
};

#endif // RecoBTag_SoftLepton_SoftLeptonAlgorithm_h
