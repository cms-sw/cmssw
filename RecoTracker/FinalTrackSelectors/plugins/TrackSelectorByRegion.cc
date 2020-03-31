#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include<vector>
#include<memory>

namespace {
  class TrackSelectorByRegion final : public edm::global::EDProducer<> {
   public:

    explicit TrackSelectorByRegion(const edm::ParameterSet& conf) :
      tracksToken_ ( consumes<reco::TrackCollection> (conf.getParameter<edm::InputTag>("tracks")) ) {
      //      for (auto const & ir : conf.getParameter<edm::VParameterSet>("regions")) {
      edm::ParameterSet regionPSet = conf.getParameter<edm::ParameterSet>("RegionPSet");
      inputTrkRegionToken_ = consumes<edm::OwnVector<TrackingRegion> >(regionPSet.getParameter<edm::InputTag>("input"));
      phiTollerance_       = regionPSet.getParameter<double>("phiTollerance");
      edm::ParameterSet trackPSet = conf.getParameter<edm::ParameterSet>("TrackPSet");
      minPt_ = trackPSet.getParameter<double>("minPt");
      //      }

      produces<reco::TrackCollection>();
      // produces<edm::RefToBaseVector<T> >();
      produces<std::vector<bool>>();

    }
      
    static void  fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("tracks",edm::InputTag("hltPixelTracks"));
      edm::ParameterSetDescription tracks_par;
      tracks_par.add<double>("minPt", 0.);
      desc.add<edm::ParameterSetDescription>("TrackPSet",tracks_par);

      edm::ParameterSetDescription region_par;
      region_par.add<edm::InputTag>("input",edm::InputTag(""));
      region_par.add<double>("phiTollerance",1.);
      desc.add<edm::ParameterSetDescription>("RegionPSet",region_par);
      descriptions.add("trackSelectorByRegion", desc);
    }

   private:

      using MaskCollection = std::vector<bool>;

      void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const override {

	// products
	auto mask = std::make_unique<MaskCollection>(); // mask w/ the same size of the input collection
	auto output_tracks = std::make_unique<reco::TrackCollection>(); // selected output collection

	edm::Handle<edm::OwnVector<TrackingRegion> > regionsHandle;
	iEvent.getByToken(inputTrkRegionToken_, regionsHandle);
       	
	edm::Handle<reco::TrackCollection> tracksHandle;
	iEvent.getByToken(tracksToken_,tracksHandle);

	std::cout << "[TrackSelectorByRegion::produce] tracksHandle.isValid ? " << (tracksHandle.isValid() ? "YEAP" : "NOPE") << std::endl;
	if ( tracksHandle.isValid() ) {
	  const auto& tracks = *tracksHandle;	  
	  mask->assign(tracks.size(),false);

	  if ( regionsHandle.isValid() ) {
	    const auto& regions = *regionsHandle;

	    const auto n_regions = regions.size();
	    std::vector<float> etaMin; etaMin.reserve(n_regions);
	    std::vector<float> etaMax; etaMax.reserve(n_regions);
	    std::vector<float> phiMin; phiMin.reserve(n_regions);
	    std::vector<float> phiMax; phiMax.reserve(n_regions);
	    std::vector<float> phi0;   phi0.reserve(n_regions);
	    std::vector<float> phi0margin; phi0margin.reserve(n_regions);
	    std::vector<math::XYZPoint> origin; origin.reserve(n_regions);
	    std::vector<float> zBound; zBound.reserve(n_regions);
	    
	    for ( const auto& tmp: regions )
	      if ( const auto *etaPhiRegion = dynamic_cast<const RectangularEtaPhiTrackingRegion *>( &tmp ) ) {
		
		const auto& etaRange  = etaPhiRegion->etaRange();
		const auto& phiMargin = etaPhiRegion->phiMargin();

		etaMin.push_back( etaRange.min() );
		etaMax.push_back( etaRange.max() );
		
		phiMin.push_back( etaPhiRegion->phiDirection() - phiMargin.left() );
		phiMax.push_back( etaPhiRegion->phiDirection() + phiMargin.right() );
		phi0.push_back( etaPhiRegion->phiDirection() );
		phi0margin.push_back( phiMargin.right() );

		GlobalPoint gp = etaPhiRegion->origin();
		origin.push_back( math::XYZPoint(gp.x(),gp.y(),gp.z()) );
		zBound.push_back( etaPhiRegion->originZBound() );
	      }


	    size_t it = 0;	    
	    std::cout << "tracks: " << tracks.size() << std::endl;
	    for ( auto trk : tracks ) {
	      
	      const auto pt = trk.pt();
	      if (pt < minPt_) {
		std::cout << " KO !!! for pt" << std::endl;
		continue;
	      }

	      const auto eta = trk.eta();
	      const auto phi = trk.phi();
	      
	      for ( size_t k=0; k<n_regions; k++ ) {
		//		if ( std::abs(trk.vz() - origin[k].z()) > zBound[k] ) {
		std::cout << "std::abs(trk.vz() - origin[k].z()): " << std::abs(trk.vz() - origin[k].z()) << " zBound: " << zBound[k] << std::endl;
		std::cout << "std::abs(trk.dz(origin[k])): " << std::abs(trk.dz(origin[k])) << " zBound: " << zBound[k] << std::endl;
		if ( std::abs(trk.dz(origin[k])) > zBound[k] ) {
		  std::cout << " KO !! for z" << std::endl;
		  continue;
		}
		std::cout << "eta : " << eta << " etaMin[k]: " << etaMin[k] << " etaMax[k]: " << etaMax[k] << std::endl;
		if ( eta < etaMin[k] ) {
		  std::cout << " KO !!! for eta" << std::endl;
		  continue;
		}
		if ( eta > etaMax[k] ) {
		  std::cout << " KO !!! for eta" << std::endl;
		  continue;
		}
		std::cout << "phi: " << phi << " phi0[k]: " << phi0[k] << " std::abs(reco::deltaPhi(phi,phi0[k])): " << std::abs(reco::deltaPhi(phi,phi0[k])) << " phi0margin: " << phi0margin[k] << std::endl;
		if ( std::abs(reco::deltaPhi(phi,phi0[k])) > phi0margin[k]*1.1 ) {
		  std::cout << " KO !!! for phi" << std::endl;
		  continue;
		}
		output_tracks->push_back(trk);
		mask->at(it) = true;
		break;
	      }
	      it++;
	    }
	  }
	  assert(mask->size()==tracks.size());
	}
	
	iEvent.put(std::move(mask));
	iEvent.put(std::move(output_tracks));
	
      }
      
      edm::EDGetTokenT<reco::TrackCollection>  tracksToken_;
      edm::EDGetTokenT<edm::OwnVector<TrackingRegion> > inputTrkRegionToken_;
      float phiTollerance_;
      float minPt_;

  };
}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackSelectorByRegion);

