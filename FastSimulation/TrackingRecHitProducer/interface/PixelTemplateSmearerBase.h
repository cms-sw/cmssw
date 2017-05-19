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
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
// template object
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate.h"

// Vectors
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"

// STL
#include <vector>
#include <string>
#include <memory>

class TFile;
class RandomEngineAndDistribution;
class SimpleHistogramGenerator;

class PixelTemplateSmearerBase:
    public TrackingRecHitAlgorithm
{
    public:
        //--- Use this type to keep track of groups of hits that need to be merged:
        struct MergeGroup{
            std::vector<TrackingRecHitProduct::SimHitIdPair> group;
            bool smearIt;
        };

    protected:
        bool mergeHitsOn; 
        std::vector< SiPixelTemplateStore > thePixelTemp_;
        int templateId;
        
        bool isFlipped(const PixelGeomDetUnit* theDet) const;
        //isForward, true for forward, false for barrel
        bool isForward;
        
        double rescotAlpha_binMin , rescotAlpha_binWidth;
        unsigned int rescotAlpha_binN;
        double rescotBeta_binMin  , rescotBeta_binWidth;
        unsigned int rescotBeta_binN;
        int resqbin_binMin, resqbin_binWidth;
        unsigned int resqbin_binN;
        

        std::map<unsigned int, const SimpleHistogramGenerator*> theXHistos;
        std::map<unsigned int, const SimpleHistogramGenerator*> theYHistos;

        std::unique_ptr<TFile> theEdgePixelResolutionFile;
        std::string theEdgePixelResolutionFileName;
        std::unique_ptr<TFile> theBigPixelResolutionFile;
        std::string theBigPixelResolutionFileName;
        std::unique_ptr<TFile> theRegularPixelResolutionFile;
        std::string theRegularPixelResolutionFileName;
        std::unique_ptr<TFile> theMergingProbabilityFile;
        std::string theMergingProbabilityFileName;
        std::unique_ptr<TFile> theMergedPixelResolutionXFile;
        std::string theMergedPixelResolutionXFileName;
        std::unique_ptr<TFile> theMergedPixelResolutionYFile;                                                                                        
        std::string theMergedPixelResolutionYFileName;

        unsigned int theLayer;

    public:

        explicit PixelTemplateSmearerBase(  const std::string& name,
			              const edm::ParameterSet& config,
			              edm::ConsumesCollector& consumesCollector );

        virtual ~PixelTemplateSmearerBase();
        virtual TrackingRecHitProductPtr process(TrackingRecHitProductPtr product) const;

        //--- Process all unmerged hits. Calls smearHit() for each.
        TrackingRecHitProductPtr processUnmergedHits( 
            std::vector<TrackingRecHitProduct::SimHitIdPair> & unmergedHits,
	        TrackingRecHitProductPtr product,
	        const PixelGeomDetUnit * detUnit,
	        const double boundX, const double boundY,
	        RandomEngineAndDistribution const * random
        ) const;
        //--- Process all groups of merged hits.
        TrackingRecHitProductPtr processMergeGroups(
            std::vector< MergeGroup* > & mergeGroups,
            TrackingRecHitProductPtr product,
            const PixelGeomDetUnit * detUnit,
            const double boundX, const double boundY,
            RandomEngineAndDistribution const * random
        ) const;


        //--- Process one umerged hit.
        FastSingleTrackerRecHit smearHit(
            const PSimHit& simHit, const PixelGeomDetUnit* detUnit, 
            const double boundX, const double boundY,
            RandomEngineAndDistribution const*) 
        const;

        //--- Process one merge group.
        FastSingleTrackerRecHit smearMergeGroup(
            MergeGroup* mg,
            const PixelGeomDetUnit * detUnit,
            const double boundX, const double boundY,
            const RandomEngineAndDistribution* random
        ) const;

        //--- Method to decide if the two hits on the same DetUnit are merged, or not.
        bool hitsMerge(const PSimHit& simHit1,const PSimHit& simHit2) const;
};
#endif
