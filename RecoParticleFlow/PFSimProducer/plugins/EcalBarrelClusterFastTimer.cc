// This producer eats standard PF ECAL clusters
// finds the corresponding fast-timing det IDs and attempts to 
// assign a reasonable time guess

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include <random>
#include <memory>

#include "SimTracker/TrackAssociation/interface/ResolutionModel.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "FWCore/Utilities/interface/isFinite.h"

class EcalBarrelClusterFastTimer : public edm::global::EDProducer<> {
public:    
  EcalBarrelClusterFastTimer(const edm::ParameterSet&);
  ~EcalBarrelClusterFastTimer() override { }
  
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  
private:
  // inputs
  edm::EDGetTokenT<EcalRecHitCollection> ebTimeHitsToken_;
  edm::EDGetTokenT<std::vector<reco::PFCluster> > ebClustersToken_;
  // options
  std::vector<std::unique_ptr<const ResolutionModel> > _resolutions;
  const float minFraction_, minEnergy_;
  const unsigned ecalDepth_;
  // functions  
  std::pair<float, DetId> getTimeForECALPFCluster(const EcalRecHitCollection&,const reco::PFCluster&) const;
  float correctTimeToVertex(const float intime, const DetId& timeDet, const reco::Vertex& vtx, 
                            const CaloSubdetectorGeometry* ecalGeom) const;
};

DEFINE_FWK_MODULE(EcalBarrelClusterFastTimer);

namespace {
  const std::string resolution("Resolution");

  constexpr float cm_per_ns = 29.9792458;

  template<typename T>
  void writeValueMap(edm::Event &iEvent,
                     const edm::Handle<std::vector<reco::PFCluster> > & handle,
                     const std::vector<T> & values,
                     const std::string    & label) {
    auto valMap = std::make_unique<edm::ValueMap<T>>();
    typename edm::ValueMap<T>::Filler filler(*valMap);
    filler.insert(handle, values.begin(), values.end());
    filler.fill();
    iEvent.put(std::move(valMap), label);
  }
}

EcalBarrelClusterFastTimer::EcalBarrelClusterFastTimer(const edm::ParameterSet& conf) :
  ebTimeHitsToken_(consumes<EcalRecHitCollection>( conf.getParameter<edm::InputTag>("ebTimeHits") ) ),
  ebClustersToken_(consumes<std::vector<reco::PFCluster> >( conf.getParameter<edm::InputTag>("ebClusters") ) ),
  minFraction_( conf.getParameter<double>("minFractionToConsider") ),
  minEnergy_(conf.getParameter<double>("minEnergyToConsider") ),
  ecalDepth_(conf.getParameter<double>("ecalDepth") )
{
  // setup resolution models
  const std::vector<edm::ParameterSet>& resos = conf.getParameterSetVector("resolutionModels");
  for( const auto& reso : resos ) {
    const std::string& name = reso.getParameter<std::string>("modelName");
    ResolutionModel* resomod = ResolutionModelFactory::get()->create(name,reso);
    _resolutions.emplace_back( resomod );  

    // times and time resolutions for general tracks
    produces<edm::ValueMap<float> >(name); 
    produces<edm::ValueMap<float> >(name+resolution);    
  }
}

void EcalBarrelClusterFastTimer::produce(edm::StreamID sid, edm::Event& evt, const edm::EventSetup& es) const {
  edm::Handle<std::vector<reco::PFCluster> > clustersH;
  edm::Handle<EcalRecHitCollection> timehitsH;

  evt.getByToken(ebClustersToken_,clustersH);
  evt.getByToken(ebTimeHitsToken_,timehitsH);
  
  const auto& clusters = *clustersH;
  const auto& timehits = *timehitsH;
  
  // get event-based seed for RNG
  unsigned int runNum_uint = static_cast <unsigned int> (evt.id().run());
  unsigned int lumiNum_uint = static_cast <unsigned int> (evt.id().luminosityBlock());
  unsigned int evNum_uint = static_cast <unsigned int> (evt.id().event());
  std::uint32_t seed = (lumiNum_uint<<10) + (runNum_uint<<20) + evNum_uint;
  std::mt19937 rng(seed);

  std::vector<std::pair<float,DetId> > times; // perfect times keyed to cluster index
  times.reserve(clusters.size());
  
  for( const auto& cluster : clusters ) {
    times.push_back( getTimeForECALPFCluster( timehits, cluster ) );
  }
  
  for( const auto& reso : _resolutions ) {
    const std::string& name = reso->name();
    std::vector<float> resolutions;
    std::vector<float> smeared_times;
    resolutions.reserve(clusters.size());
    smeared_times.reserve(clusters.size());
    
    // smear once then correct to multiple vertices
    for( unsigned i = 0 ; i < clusters.size(); ++i ) {      
      const float theresolution = reso->getTimeResolution(clusters[i]);
      std::normal_distribution<float> gausTime(times[i].first, theresolution);
      
      smeared_times.emplace_back( gausTime(rng) );
      resolutions.push_back( theresolution );
    }    

    writeValueMap(evt,clustersH,smeared_times,name);
    writeValueMap(evt,clustersH,resolutions,name+resolution);
  }

}

std::pair<float,DetId> EcalBarrelClusterFastTimer::getTimeForECALPFCluster(const EcalRecHitCollection& timehits, const reco::PFCluster& cluster) const {
  
  const auto& rhfs = cluster.recHitFractions();  
  unsigned best_hit = 0;
  float best_energy = -1.f;
  for( const auto& rhf : rhfs ) {
    const auto& hitref = rhf.recHitRef();
    const unsigned detid = hitref->detId();
    const auto fraction = rhf.fraction();
    const auto energy = hitref->energy();
    if( fraction < minFraction_ || energy < minEnergy_ ) continue;
    auto timehit = timehits.find(detid);
    float e_hit = rhf.fraction() * hitref->energy();
    if( e_hit > best_energy && timehit->isTimeValid() ) {
      best_energy = e_hit;
      best_hit = detid;
    }
  }
  
  float best_time_guess = std::numeric_limits<float>::max();
  if( best_energy > 0. ) {
    best_time_guess = timehits.find(best_hit)->time();
  }
  
  //std::cout << "EcalBarrelFastTimer: " << best_time_guess << ' ' << best_energy << ' ' << best_hit << std::endl;

  return std::make_pair(best_time_guess,DetId(best_hit));
}

float EcalBarrelClusterFastTimer::correctTimeToVertex(const float intime, const DetId& timeDet, const reco::Vertex& vtx, 
                                                      const CaloSubdetectorGeometry* ecalGeom) const {
  if( timeDet.rawId() == 0 ) return -999.;
  // correct the cluster time from 0,0,0 to the primary vertex given
  auto cellGeometry = ecalGeom->getGeometry(timeDet);
  if( nullptr == cellGeometry ) {
    throw cms::Exception("BadECALBarrelCell")
      << std::hex << timeDet.rawId() << std::dec<< " is not a valid ECAL Barrel DetId!";
  }
  //depth in mm in the middle of the layer position;
  GlobalPoint layerPos = cellGeometry->getPosition( ecalDepth_+0.5 ); 
  const math::XYZPoint layerPos_cm( layerPos.x(), layerPos.y(), layerPos.z() );
  const math::XYZVector to_center = layerPos_cm - math::XYZPoint(0.,0.,0.);
  const math::XYZVector to_vtx = layerPos_cm - vtx.position();
  
  /*
  std::cout << intime << ' ' << to_center.r()/cm_per_ns << ' ' << to_vtx.r()/cm_per_ns
            << ' ' << intime + to_center.r()/cm_per_ns - to_vtx.r()/cm_per_ns << std::endl;
  */

  return intime + to_center.r()/cm_per_ns - to_vtx.r()/cm_per_ns;
}
