#ifndef CandidateSeededTrackingRegionsProducer_h
#define CandidateSeededTrackingRegionsProducer_h

// $Id: CandidateSeededTrackingRegionsProducer.h,v 1.1 2012/03/13 16:20:53 khotilov Exp $

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

/** class CandidateSeededTrackingRegionsProducer
 *
 * eta-phi TrackingRegions producer in directions defined by Candidate-based objects of interest
 * from a collection defined by the "input" parameter.
 *
 * Four operational modes are supported ("mode" parameter):
 *
 *   BeamSpotFixed:
 *     origin is defined by the beam spot
 *     z-half-length is defined by a fixed zErrorBeamSpot parameter
 *   BeamSpotSigma:
 *     origin is defined by the beam spot
 *     z-half-length is defined by nSigmaZBeamSpot * beamSpot.sigmaZ
 *   VerticesFixed:
 *     origins are defined by vertices from VertexCollection (use maximum MaxNVertices of them)
 *     z-half-length is defined by a fixed zErrorVetex parameter
 *   VerticesSigma:
 *     origins are defined by vertices from VertexCollection (use maximum MaxNVertices of them)
 *     z-half-length is defined by nSigmaZVertex * vetex.zError
 *
 *   If, while using one of the "Vertices" modes, there's no vertices in an event, we fall back into
 *   either BeamSpotSigma or BeamSpotFixed mode, depending on the positiveness of nSigmaZBeamSpot.
 *
 *   \author Vadim Khotilovich
 */
class  CandidateSeededTrackingRegionsProducer : public TrackingRegionProducer
{
public:

  typedef enum {BEAM_SPOT_FIXED, BEAM_SPOT_SIGMA, VERTICES_FIXED, VERTICES_SIGMA } Mode;

  explicit CandidateSeededTrackingRegionsProducer(const edm::ParameterSet& conf)
  {
    edm::ParameterSet regPSet = conf.getParameter<edm::ParameterSet>("RegionPSet");

    // operation mode
    std::string modeString       = regPSet.getParameter<std::string>("mode");
    if      (modeString == "BeamSpotFixed") m_mode = BEAM_SPOT_FIXED;
    else if (modeString == "BeamSpotSigma") m_mode = BEAM_SPOT_SIGMA;
    else if (modeString == "VerticesFixed") m_mode = VERTICES_FIXED;
    else if (modeString == "VerticesSigma") m_mode = VERTICES_SIGMA;
    else  edm::LogError ("CandidateSeededTrackingRegionsProducer")<<"Unknown mode string: "<<modeString;

    // basic inputs
    m_input            = regPSet.getParameter<edm::InputTag>("input");
    m_maxNRegions      = regPSet.getParameter<int>("maxNRegions");
    m_beamSpot         = regPSet.getParameter<edm::InputTag>("beamSpot");
    m_vertexCollection = edm::InputTag();
    m_maxNVertices     = 1;
    if (m_mode == VERTICES_FIXED || m_mode == VERTICES_SIGMA)
    {
      m_vertexCollection = regPSet.getParameter<edm::InputTag>("vertexCollection");
      m_maxNVertices     = regPSet.getParameter<int>("maxNVertices");
    }

    // RectangularEtaPhiTrackingRegion parameters:
    m_ptMin            = regPSet.getParameter<double>("ptMin");
    m_originRadius     = regPSet.getParameter<double>("originRadius");
    m_zErrorBeamSpot   = regPSet.getParameter<double>("zErrorBeamSpot");
    m_deltaEta         = regPSet.getParameter<double>("deltaEta");
    m_deltaPhi         = regPSet.getParameter<double>("deltaPhi");
    m_precise          = regPSet.getParameter<bool>("precise");
    m_measurementTrackerName       = "";
    m_whereToUseMeasurementTracker = 0;
    if (regPSet.exists("measurementTrackerName"))
    {
      m_measurementTrackerName = regPSet.getParameter<std::string>("measurementTrackerName");
      if (regPSet.exists("whereToUseMeasurementTracker"))
        m_whereToUseMeasurementTracker = regPSet.getParameter<double>("whereToUseMeasurementTracker");
    }
    m_searchOpt = false;
    if (regPSet.exists("searchOpt")) m_searchOpt = regPSet.getParameter<bool>("searchOpt");

    // mode-dependent z-halflength of tracking regions
    if (m_mode == VERTICES_SIGMA)  m_nSigmaZVertex   = regPSet.getParameter<double>("nSigmaZVertex");
    if (m_mode == VERTICES_FIXED)  m_zErrorVetex     = regPSet.getParameter<double>("zErrorVetex");
    m_nSigmaZBeamSpot = -1.;
    if (m_mode == BEAM_SPOT_SIGMA)
    {
      m_nSigmaZBeamSpot = regPSet.getParameter<double>("nSigmaZBeamSpot");
      if (m_nSigmaZBeamSpot < 0.)
        edm::LogError ("CandidateSeededTrackingRegionsProducer")<<"nSigmaZBeamSpot must be positive for BeamSpotSigma mode!";
    }
  }
  
  virtual ~CandidateSeededTrackingRegionsProducer() {}
    

  virtual std::vector<TrackingRegion* > regions(const edm::Event& e, const edm::EventSetup& es) const
  {
    std::vector<TrackingRegion* > result;

    // pick up the candidate objects of interest
    edm::Handle< reco::CandidateView > objects;
    e.getByLabel( m_input, objects );
    size_t n_objects = objects->size();
    if (n_objects == 0) return result;

    // always need the beam spot (as a fall back strategy for vertex modes)
    edm::Handle< reco::BeamSpot > bs;
    e.getByLabel( m_beamSpot, bs );
    if( !bs.isValid() ) return result;

    // this is a default origin for all modes
    GlobalPoint default_origin( bs->x0(), bs->y0(), bs->z0() );

    // vector of origin & halfLength pairs:
    std::vector< std::pair< GlobalPoint, float > > origins;

    // fill the origins and halfLengths depending on the mode
    if (m_mode == BEAM_SPOT_FIXED || m_mode == BEAM_SPOT_SIGMA)
    {
      origins.push_back( std::make_pair(
          default_origin,
          (m_mode == BEAM_SPOT_FIXED) ? m_zErrorBeamSpot : m_nSigmaZBeamSpot*bs->sigmaZ()
      ));
    }
    else if (m_mode == VERTICES_FIXED || m_mode == VERTICES_SIGMA)
    {
      edm::Handle< reco::VertexCollection > vertices;
      e.getByLabel( m_vertexCollection, vertices );
      int n_vert = 0;
      for (reco::VertexCollection::const_iterator v = vertices->begin(); v != vertices->end() && n_vert < m_maxNVertices; ++v)
      {
        if ( v->isFake() || !v->isValid() ) continue;

        origins.push_back( std::make_pair(
            GlobalPoint( v->x(), v->y(), v->z() ),
            (m_mode == VERTICES_FIXED) ? m_zErrorVetex : m_nSigmaZVertex*v->zError()
        ));
        ++n_vert;
      }
      // no-vertex fall-back case:
      if (origins.empty())
      {
        origins.push_back( std::make_pair(
            default_origin,
            (m_nSigmaZBeamSpot > 0.) ? m_nSigmaZBeamSpot*bs->z0Error() : m_zErrorBeamSpot
        ));
      }
    }
    
    // create tracking regions (maximum MaxNRegions of them) in directions of the
    // objects of interest (we expect that the collection was sorted in decreasing pt order)
    int n_regions = 0;
    for(size_t i = 0; i < n_objects && n_regions < m_maxNRegions; ++i )
    {
      const reco::Candidate & object = (*objects)[i];
      GlobalVector direction( object.momentum().x(), object.momentum().y(), object.momentum().z() );

      for (size_t  j=0; j<origins.size() && n_regions < m_maxNRegions; ++j)
      {
        result.push_back( new RectangularEtaPhiTrackingRegion(
          direction,
          origins[j].first,
          m_ptMin,
          m_originRadius,
          origins[j].second,
          m_deltaEta,
          m_deltaPhi,
          m_whereToUseMeasurementTracker,
          m_precise,
          m_measurementTrackerName,
          m_searchOpt
        ));
        ++n_regions;
      }
    }
    //std::cout<<"n_seeded_regions = "<<n_regions<<std::endl;
    edm::LogInfo ("CandidateSeededTrackingRegionsProducer") << "produced "<<n_regions<<" regions";

    return result;
  }
  
private:

  Mode m_mode;

  edm::InputTag m_input;
  int m_maxNRegions;
  edm::InputTag m_beamSpot;
  edm::InputTag m_vertexCollection;
  int m_maxNVertices;

  float m_ptMin;
  float m_originRadius;
  float m_zErrorBeamSpot;
  float m_deltaEta;
  float m_deltaPhi;
  bool m_precise;
  std::string m_measurementTrackerName;
  float m_whereToUseMeasurementTracker;
  bool m_searchOpt;

  float m_nSigmaZVertex;
  float m_zErrorVetex;
  float m_nSigmaZBeamSpot;
};

#endif
