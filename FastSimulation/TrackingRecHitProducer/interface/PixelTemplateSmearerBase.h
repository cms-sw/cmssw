#ifndef FastSimulation_TrackingRecHitProducer_PixelTemplateSmearerBase_h
#define FastSimulation_TrackingRecHitProducer_PixelTemplateSmearerBase_h

//---------------------------------------------------------------------------
//! \class SiTrackerGaussianSmearingRecHits
//!
//! \brief EDProducer to create RecHits from PSimHits with Gaussian smearing
//!
//---------------------------------------------------------------------------

// FastSim stuff
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"

//Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// PSimHit
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

// Geometry
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
// template object
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"

// Vectors
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"

// STL.  <memory> needed for uniq_ptr<>
#include <vector>
#include <string>
#include <memory>

class TFile;
class RandomEngineAndDistribution;
class SimpleHistogramGenerator;
class PixelResolutionHistograms;

class PixelTemplateSmearerBase : public TrackingRecHitAlgorithm {
public:
  //--- Use this type to keep track of groups of hits that need to be merged:
  struct MergeGroup {
    std::vector<TrackingRecHitProduct::SimHitIdPair> group;
    bool smearIt;
  };

protected:
  bool mergeHitsOn = false;  // if true then see if neighboring hits might merge

  //--- Template DB Object(s)
  const SiPixelTemplateDBObject* pixelTemplateDBObject_ = nullptr;            // needed for template<-->DetId map.
  std::vector<SiPixelTemplateStore> thePixelTemp_;                            // our own template storage
  const std::vector<SiPixelTemplateStore>* thePixelTempRef = &thePixelTemp_;  // points to the one we will use.
  int templateId = -1;

  //--- Flag to tell us whether we are in barrel or in forward.
  //    This is needed since the parameterization is slightly
  //    different for forward, since all forward detectors cover
  //    a smaller range of local incidence angles and thus
  //    the clusters are shorter and have less charge.
  bool isBarrel;

  //--- The histogram storage containers.
  std::shared_ptr<PixelResolutionHistograms> theEdgePixelResolutions;
  std::string theEdgePixelResolutionFileName;

  std::shared_ptr<PixelResolutionHistograms> theBigPixelResolutions;
  std::string theBigPixelResolutionFileName;

  std::shared_ptr<PixelResolutionHistograms> theRegularPixelResolutions;
  std::string theRegularPixelResolutionFileName;

  //--- Files with hit merging information:
  std::unique_ptr<TFile> theMergingProbabilityFile;
  std::string theMergingProbabilityFileName;

  std::unique_ptr<TFile> theMergedPixelResolutionXFile;
  std::string theMergedPixelResolutionXFileName;

  std::unique_ptr<TFile> theMergedPixelResolutionYFile;
  std::string theMergedPixelResolutionYFileName;

public:
  explicit PixelTemplateSmearerBase(const std::string& name,
                                    const edm::ParameterSet& config,
                                    edm::ConsumesCollector& consumesCollector);

  ~PixelTemplateSmearerBase() override;
  TrackingRecHitProductPtr process(TrackingRecHitProductPtr product) const override;
  // void beginEvent(edm::Event& event, const edm::EventSetup& eventSetup) override;
  void beginRun(edm::Run const& run,
                const edm::EventSetup& eventSetup,
                const SiPixelTemplateDBObject* pixelTemplateDBObjectPtr,
                const std::vector<SiPixelTemplateStore>& tempStoreRef) override;
  // void endEvent(edm::Event& event, const edm::EventSetup& eventSetup) override;

  //--- Process all unmerged hits. Calls smearHit() for each.
  TrackingRecHitProductPtr processUnmergedHits(std::vector<TrackingRecHitProduct::SimHitIdPair>& unmergedHits,
                                               TrackingRecHitProductPtr product,
                                               const PixelGeomDetUnit* detUnit,
                                               const double boundX,
                                               const double boundY,
                                               RandomEngineAndDistribution const* random) const;
  //--- Process all groups of merged hits.
  TrackingRecHitProductPtr processMergeGroups(std::vector<MergeGroup*>& mergeGroups,
                                              TrackingRecHitProductPtr product,
                                              const PixelGeomDetUnit* detUnit,
                                              const double boundX,
                                              const double boundY,
                                              RandomEngineAndDistribution const* random) const;

  //--- Process one umerged hit.
  FastSingleTrackerRecHit smearHit(const PSimHit& simHit,
                                   const PixelGeomDetUnit* detUnit,
                                   const double boundX,
                                   const double boundY,
                                   RandomEngineAndDistribution const*) const;

  //--- Process one merge group.
  FastSingleTrackerRecHit smearMergeGroup(MergeGroup* mg,
                                          const PixelGeomDetUnit* detUnit,
                                          const double boundX,
                                          const double boundY,
                                          const RandomEngineAndDistribution* random) const;

  //--- Method to decide if the two hits on the same DetUnit are merged, or not.
  bool hitsMerge(const PSimHit& simHit1, const PSimHit& simHit2) const;
};
#endif
