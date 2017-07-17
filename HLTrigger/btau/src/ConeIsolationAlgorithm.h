#ifndef ConeIsolationAlgorithm_H
#define ConeIsolationAlgorithm_H
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class  ConeIsolationAlgorithm  {
public:
 
  ConeIsolationAlgorithm(const edm::ParameterSet  & parameters );
  ConeIsolationAlgorithm(); 

  // For out of framework usage we may need a different constructor
  // so we keep datamember as builtin types (instead of ParameterSet) 
  // ConeIsolationAlgorithm (int,float,....);
   
  ~ConeIsolationAlgorithm() {}

  static void fillDescription(edm::ParameterSetDescription& desc);

  std::pair<float ,reco::IsolatedTauTagInfo> tag( const reco::JetTracksAssociationRef & jetTracks, const reco::Vertex & pv); 
  

 private:
  // algorithm parameters
  int    m_cutPixelHits;
  int    m_cutTotalHits;
  double m_cutMaxTIP;
  double m_cutMinPt;
  double m_cutMaxChiSquared;
  double matching_cone;
  double signal_cone;
  double isolation_cone;
  double pt_min_isolation;
  double pt_min_leadTrack;
  double dZ_vertex;
  int    n_tracks_isolation_ring;
  bool   useVertexConstrain_;
  bool   useFixedSizeCone;
  double variableConeParameter;
  double variableMaxCone;
  double variableMinCone;
};

#endif // ConeIsolationAlgorithm_H
