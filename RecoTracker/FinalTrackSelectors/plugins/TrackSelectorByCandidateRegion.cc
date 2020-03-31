#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include<vector>
#include<memory>

namespace {
  class TrackSelectorByCandidateRegion final : public edm::global::EDProducer<> {
   public:

    typedef enum {BEAM_SPOT_FIXED, BEAM_SPOT_SIGMA, VERTICES_FIXED, VERTICES_SIGMA } Mode;

    explicit TrackSelectorByCandidateRegion(const edm::ParameterSet& conf) :
      tracksToken_  ( consumes<reco::TrackCollection> (conf.getParameter<edm::InputTag>("tracks"))   ),
      beamspotToken_( consumes<reco::BeamSpot>        (conf.getParameter<edm::InputTag>("beamspot")) ) {

      edm::ParameterSet trackPSet = conf.getParameter<edm::ParameterSet>("TrackPSet");
      minPt_ = trackPSet.getParameter<double>("minPt");
      edm::ParameterSet chi2_vs_pt = trackPSet.getParameter<edm::ParameterSet>("chi2vsPt");
      bool enabled = chi2_vs_pt.getParameter<bool>("enabled");
      pt1_ = enabled ? chi2_vs_pt.getParameter<double>("pt1") : -1.;
      pt2_ = enabled ? chi2_vs_pt.getParameter<double>("pt2") : -1.;
      value1_ = enabled ? chi2_vs_pt.getParameter<double>("value1") : -1.;
      value2_ = enabled ? chi2_vs_pt.getParameter<double>("value2") : -1.;

      if (enabled && pt1_ >= pt2_)
	throw cms::Exception("Configuration") << "TrackSelectorByCandidateRegion::chi2vsPt: pt1 (" << pt1_ << ") needs to be smaller than pt2 (" << pt2_ << ")";
      if (enabled && pt1_ <= 0)
	throw cms::Exception("Configuration") << "TrackSelectorByCandidateRegion::chi2vsPt: pt1 needs to be > 0; is " << pt1_;
      if (enabled && pt2_ <= 0)
	throw cms::Exception("Configuration") << "TrackSelectorByCandidateRegion::chi2vsPt: pt2 needs to be > 0; is " << pt2_;

      //      for (auto const & ir : conf.getParameter<edm::VParameterSet>("regions")) {
      edm::ParameterSet regionPSet = conf.getParameter<edm::ParameterSet>("RegionPSet");

      componentName_ = regionPSet.getParameter<std::string>("componentName");
      /**
	 eta-phi TrackingRegions producer in directions defined by Candidate-based objects of interest
	 from a collection defined by the "input" parameter.
	 Four operational modes are supported ("mode" parameter):
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
	 **/
      
	inputCandidateToken_ = mayConsume<reco::CandidateView>(regionPSet.getParameter<edm::InputTag>("input"));
	maxNRegions_ = regionPSet.getParameter<unsigned int>("maxNRegions");
	
	std::string mode = regionPSet.getParameter<std::string>("mode");
	if      (mode == "BeamSpotFixed") mode_ = BEAM_SPOT_FIXED;
	else if (mode == "BeamSpotSigma") mode_ = BEAM_SPOT_SIGMA;
	else if (mode == "VerticesFixed") mode_ = VERTICES_FIXED;
	else if (mode == "VerticesSigma") mode_ = VERTICES_SIGMA;
	//	else  edm::LogError ("TrackSelectorByCandidateRegion") << "Unknown mode string: " << mode;

	maxNVertices_    = 1;
	if (mode == VERTICES_FIXED || mode == VERTICES_SIGMA) {
	  verticesToken_ = mayConsume<reco::VertexCollection>(regionPSet.getParameter<edm::InputTag>("vertices"));
	  maxNVertices_   = regionPSet.getParameter<int>("maxNVertices");
	}
	
	// RectangularEtaPhiTrackingRegion parameters:
	deltaEta_         = regionPSet.getParameter<double>("deltaEta");
	deltaPhi_         = regionPSet.getParameter<double>("deltaPhi");
	zErrorBeamSpot_   = regionPSet.getParameter<double>("zErrorBeamSpot");
	// mode-dependent z-halflength of tracking regions
	nSigmaZVertex_   = ( mode_ == VERTICES_SIGMA   ? regionPSet.getParameter<double>("nSigmaZVertex")   : -1. );
	zErrorVetex_     = ( mode_ == VERTICES_FIXED   ? regionPSet.getParameter<double>("zErrorVetex")     : -1. );
	nSigmaZBeamSpot_ = ( mode_ == BEAM_SPOT_SIGMA  ? regionPSet.getParameter<double>("nSigmaZBeamSpot") : -1. );
	//      }

	produces<reco::TrackCollection>();
	// produces<edm::RefToBaseVector<T> >();
	produces<std::vector<bool>>();
	
    }
      
    static void  fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("tracks",  edm::InputTag("hltPixelTracks"));
      desc.add<edm::InputTag>("beamspot",edm::InputTag("hltOnlineBeamSpot"));
      edm::ParameterSetDescription tracks_par;
      tracks_par.add<double>("minPt", 0.);
      edm::ParameterSetDescription chi2vsPt_par;
      chi2vsPt_par.add<bool>("enabled",false);
      chi2vsPt_par.add<double>("pt1",-1.);
      chi2vsPt_par.add<double>("pt2",-1.);
      chi2vsPt_par.add<double>("value1",-1.);
      chi2vsPt_par.add<double>("value2",-1.);
      tracks_par.add<edm::ParameterSetDescription>("chi2vsPt",chi2vsPt_par);
      desc.add<edm::ParameterSetDescription>("TrackPSet",tracks_par);

      edm::ParameterSetDescription region_par;
      region_par.add<std::string>("componentName", "");
      region_par.add<edm::InputTag>("input",   edm::InputTag(""));
      region_par.add<unsigned int>("maxNRegions", 10);
      region_par.add<std::string>("mode",    "");
      region_par.add<edm::InputTag>("vertices",edm::InputTag(""));
      region_par.add<int>("maxNVertices",3);
      region_par.add<double>("deltaEta", 0.5);
      region_par.add<double>("deltaPhi", 0.5);
      region_par.add<double>("zErrorBeamSpot", -1.);
      region_par.add<double>("nSigmaZVertex",  -1.);
      region_par.add<double>("zErrorVetex",    -1.);
      region_par.add<double>("nSigmaZBeamSpot",-1.);
      desc.add<edm::ParameterSetDescription>("RegionPSet",region_par);
      descriptions.add("trackSelectorByCandidateRegion", desc);
    }

   private:

      using MaskCollection = std::vector<bool>;

      void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& eSetup) const override {

	// products
	auto mask = std::make_unique<MaskCollection>(); // mask w/ the same size of the input collection
	auto output_tracks = std::make_unique<reco::TrackCollection>(); // selected output collection

	edm::Handle< reco::CandidateView > objectsHandle;
	// pick up the candidate objects of interest	
	iEvent.getByToken( inputCandidateToken_, objectsHandle );
	
	// always need the beam spot (as a fall back strategy for vertex modes)
	edm::Handle< reco::BeamSpot > beamspotHandle;
	iEvent.getByToken( beamspotToken_, beamspotHandle );
	
	edm::Handle<reco::TrackCollection> tracksHandle;
	iEvent.getByToken(tracksToken_,tracksHandle);

	if ( tracksHandle.isValid() ) {
	  const auto& tracks = *tracksHandle;	  
	  mask->assign(tracks.size(),false);

	  if ( objectsHandle.isValid() && beamspotHandle.isValid() ) {
	    
	    const auto& beamspot = *beamspotHandle;
	    // this is a default origin for all modes
	    math::XYZPoint default_origin( beamspot.x0(), beamspot.y0(), beamspot.z0() );
	    
	    // vector of origin & halfLength pairs:
	    std::vector< std::pair< math::XYZPoint, float > > origins;
	    
	    // fill the origins and halfLengths depending on the mode
	    if (mode_ == BEAM_SPOT_FIXED) 
	      origins.push_back( std::make_pair(default_origin, zErrorBeamSpot_) );
	    else if (mode_ == BEAM_SPOT_SIGMA)
	      origins.push_back( std::make_pair(default_origin, nSigmaZBeamSpot_*beamspot.sigmaZ()) );
	    else if (mode_ == VERTICES_FIXED || mode_ == VERTICES_SIGMA) {
	      
	      edm::Handle< reco::VertexCollection > verticesHandle;
	      iEvent.getByToken( verticesToken_, verticesHandle );
	      if ( verticesHandle.isValid() ) {	      
		int n_vert = 0;
		const auto& vertices = *verticesHandle;
		for ( auto v : vertices ) {
		  if ( n_vert > maxNVertices_ ) break;
		  if ( v.isFake() || !v.isValid() ) continue;
		  if (mode_ == VERTICES_FIXED)
		    origins.push_back( std::make_pair(math::XYZPoint( v.x(), v.y(), v.z() ), zErrorVetex_ ) );
		  else if (mode_ == VERTICES_SIGMA)
		    origins.push_back( std::make_pair(math::XYZPoint( v.x(), v.y(), v.z() ), nSigmaZVertex_*v.zError()) );
		  ++n_vert;
		}
	      }
	      // no-vertex fall-back case:
	      if (origins.empty())
		origins.push_back( std::make_pair(default_origin, nSigmaZBeamSpot_*beamspot.z0Error()) );
	    }

	    std::vector<float> etaMin; etaMin.reserve(maxNRegions_);
	    std::vector<float> etaMax; etaMax.reserve(maxNRegions_);
	    std::vector<float> phiMin; phiMin.reserve(maxNRegions_);
	    std::vector<float> phiMax; phiMax.reserve(maxNRegions_);
	    std::vector<math::XYZPoint> origin; origin.reserve(maxNRegions_);
	    std::vector<float> zBound;       zBound.reserve(maxNRegions_);

	    size_t n_objects = objectsHandle->size();
	    size_t n_regions = 0;
	    for ( size_t i = 0; i < n_objects && n_regions < maxNRegions_; ++i ) {
	      std::cout << "i: " << i << " n_regions: " << n_regions << std::endl;
	      const reco::Candidate & object = (*objectsHandle)[i];
	      //	  GlobalVector direction( object.momentum().x(), object.momentum().y(), object.momentum().z() );
	      for ( size_t j = 0; j < origins.size() && n_regions < maxNRegions_; ++j ) {
		etaMin.push_back( object.eta() - deltaEta_ );
		etaMax.push_back( object.eta() + deltaEta_ );
		phiMin.push_back( Geom::Phi( object.phi() - deltaPhi_ ) );
		phiMax.push_back( Geom::Phi( object.phi() + deltaPhi_ ) );
		origin.push_back( origins[j].first );
		zBound.push_back( origins[j].second );

		n_regions++;
	      }
	    }

	    size_t it = 0;
	    const auto curvature1 = (value1_ >= 0.) ? PixelRecoUtilities::curvature(1.f / pt1_, eSetup) : -1.;
	    const auto curvature2 = (value2_ >= 0.) ? PixelRecoUtilities::curvature(1.f / pt2_, eSetup) : -1.;
	    for ( auto trk : tracks ) {
	      
	      const auto pt = trk.pt();
	      if (pt < minPt_) continue;
	      if (value1_ >= 0.) {
		const auto curvature = PixelRecoUtilities::curvature(1.f / pt, eSetup);
		// https://cmssdt.cern.ch/dxr/CMSSW/source/RecoPixelVertexing/PixelTriplets/interface/CAHitQuadrupletGenerator.h#58		
		auto value = -1.;
		if (curvature1 < curvature)
		  value = value1_;
		else if (curvature2 < curvature && curvature <= curvature1)
		  value = value2_ + (curvature - curvature2) / (curvature1 - curvature2) * (value1_ - value2_);
		else 
		  value = value2_;

		const auto chi2 = trk.chi2();
		if (chi2 > value) continue;
	      }

	      const auto eta = trk.eta();
	      const auto phi = trk.phi();

	      for ( size_t k = 0; k < n_regions; ++k ) {
		if ( fabs(trk.dz(origin[k])) > zBound[k] ) continue;
		//		if ( fabs(trk.vz() - origin[k].z()) > zBound[k] ) continue;
		
		if ( (eta >= etaMin[k] && eta <= etaMax[k]) &&
		     (phi >= phiMin[k] && phi <= phiMax[k]) ) {
		  
		  output_tracks->push_back(trk);
		  mask->at(it) = true;
		  break;
		}
	      }
	      ++it;
	    }
	  }
	  assert(mask->size()==tracks.size());
	}

	iEvent.put(std::move(mask));
	iEvent.put(std::move(output_tracks));
      }
      
      Mode mode_;
      
      edm::EDGetTokenT<reco::TrackCollection>  tracksToken_;
      edm::EDGetTokenT<reco::CandidateView>    inputCandidateToken_; 
      edm::EDGetTokenT<reco::VertexCollection> verticesToken_;
      edm::EDGetTokenT<reco::BeamSpot>         beamspotToken_;

      float minPt_;
      float pt1_;
      float pt2_;
      float value1_;
      float value2_;

      size_t maxNRegions_;
      int maxNVertices_;
      std::string componentName_;
      float zErrorBeamSpot_;
      float deltaEta_;
      float deltaPhi_;
      float nSigmaZVertex_;
      float zErrorVetex_;
      float nSigmaZBeamSpot_;

  };
}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackSelectorByCandidateRegion);

