

#include <memory>
#include <vector>
#include <functional>
#include <array>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
//#include "RecoPixelVertexing/PixelTriplets/plugins/LayerTriplets.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitSeeds.h"

class TrackerTopology;
class SeedCreator;
namespace edm { class ConsumesCollector; }

///
/// helper class for extracting info
/// from layer bare names
///
class SeedMergerPixelLayer {

 public:
  enum Side{ Minus=1, Plus=2, Undefined, SideError }; // Changed to correspond with PXFDetId ...

  SeedMergerPixelLayer( const std::string& );
  unsigned getLayerNumber( void ) const { return layer_; }
  Side getSide( void ) const { return side_;}
  PixelSubdetector::SubDetector getSubdet( void ) const { return subdet_; }
  std::string getName( void ) const { return name_; }
  bool isContainsDetector( const DetId&, const TrackerTopology *tTopo ) const;

 private:
  bool isValidName( const std::string& );
  bool isValid_;
  std::string name_;
  PixelSubdetector::SubDetector subdet_;
  unsigned layer_;
  SeedMergerPixelLayer::Side side_;

};



///
/// merge triplets into quadruplets
///
class QuadrupletSeedMerger {

 public:

  explicit QuadrupletSeedMerger(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC);
  QuadrupletSeedMerger(const edm::ParameterSet& iConfig, const edm::ParameterSet& seedCreatorConfig, edm::ConsumesCollector& iC);
  ~QuadrupletSeedMerger();

  void update(const edm::EventSetup& );
  //const std::vector<SeedingHitSet> mergeTriplets( const OrderedSeedingHits&, const edm::EventSetup& );
  const OrderedSeedingHits& mergeTriplets( const OrderedSeedingHits&, const edm::EventSetup& );
  const TrajectorySeedCollection mergeTriplets( const TrajectorySeedCollection&, const TrackingRegion&, const edm::EventSetup&);
  std::pair<double,double> calculatePhiEta( SeedingHitSet const& ) const;
  void printHit( const TrackingRecHit* ) const;
  void printNtuplet( const SeedingHitSet& ) const;
  void setMergeTriplets( bool );
  void setAddRemainingTriplets( bool );
  void setTTRHBuilderLabel( std::string );

 private:
  typedef std::array<SeedingHitSet::ConstRecHitPointer, 4> QuadrupletHits;

  QuadrupletSeedMerger(const edm::ParameterSet& iConfig, SeedCreator *seedCreator, edm::ConsumesCollector& iC);

  void mySort(QuadrupletHits& unsortedHits);

  bool isValidQuadruplet(const QuadrupletHits& quadruplet, const std::vector<SeedMergerPixelLayer>& layers, const TrackerTopology *tTopo) const;

    // bool isValidQuadruplet( const SeedingHitSet&, const std::vector<SeedMergerPixelLayer>& ) const;

  SeedingLayerSetsBuilder theLayerBuilder_;
  ctfseeding::SeedingLayerSets theLayerSets_;
  edm::ESHandle<TrackerGeometry> theTrackerGeometry_;
  edm::ESHandle<TransientTrackingRecHitBuilder> theTTRHBuilder_;
  std::unique_ptr<SeedCreator> theSeedCreator_;
  bool isMergeTriplets_;
  bool isAddRemainingTriplets_;
  std::string theTTRHBuilderLabel_;
  OrderedHitSeeds quads_; 
};
