#ifndef TrackProbabilityAlgorithm_H
#define TrackProbabilityAlgorithm_H
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTag/TrackProbability/interface/HistogramProbabilityEstimator.h"

class TransientTrackBuilder;

class  TrackProbabilityAlgorithm  {

public:
 
  TrackProbabilityAlgorithm(const edm::ParameterSet  & parameters );
  TrackProbabilityAlgorithm(); 

  // For out of framework usage we may need a different constructor
  // so we keep datamember as builtin types (instead of ParameterSet) 
  //TrackProbabilityAlgorithm (int,float,....);
   
  ~TrackProbabilityAlgorithm() {}

   void setTransientTrackBuilder(const TransientTrackBuilder * builder) { m_transientTrackBuilder = builder; }
   void setProbabilityEstimator(HistogramProbabilityEstimator * esti) { m_probabilityEstimator = esti; }
   HistogramProbabilityEstimator * probabilityEstimator() {return m_probabilityEstimator; } 
  std::pair<reco::JetTag,reco::TrackProbabilityTagInfo> tag(const  reco::JetTracksAssociationRef & jetTracks, const reco::Vertex & pv); 
  

 private:
 // reco::Vertex * m_primaryVertex;
 //const  MagneticField * m_magneticField;

 const TransientTrackBuilder * m_transientTrackBuilder;
 HistogramProbabilityEstimator * m_probabilityEstimator;
 
//algorithm parameters

int  m_ipType;
unsigned int  m_cutPixelHits;
unsigned int  m_cutTotalHits;
double  m_cutMaxTIP;
double  m_cutMinPt;
double  m_cutMaxDecayLen;
double  m_cutMaxChiSquared;
double  m_cutMaxLIP;
double m_cutMaxDistToAxis;
double m_cutMinProb;
};

#endif 

