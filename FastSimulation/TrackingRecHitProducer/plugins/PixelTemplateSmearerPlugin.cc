
// SiPixel Gaussian Smearing
#include "FastSimulation/TrackingRecHitProducer/interface/PixelTemplateSmearerBase.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithmFactory.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"
#include "FastSimulation/TrackingRecHitProducer/interface/PixelResolutionHistograms.h"

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
/// If we ever need to port back to 9X: #include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"

// Famos
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/Utilities/interface/SimpleHistogramGenerator.h"


class PixelTemplateSmearerPlugin:
  public PixelTemplateSmearerBase
{
public:
  explicit PixelTemplateSmearerPlugin( const std::string& name,
				       const edm::ParameterSet& config,
				       edm::ConsumesCollector& consumesCollector
				       );
  ~PixelTemplateSmearerPlugin() override;
};


PixelTemplateSmearerPlugin::PixelTemplateSmearerPlugin(
    const std::string& name,
    const edm::ParameterSet& config,
    edm::ConsumesCollector& consumesCollector
):
  PixelTemplateSmearerBase(name, config, consumesCollector)
{
}


PixelTemplateSmearerPlugin::~PixelTemplateSmearerPlugin()
{
}


DEFINE_EDM_PLUGIN(
    TrackingRecHitAlgorithmFactory,
    PixelTemplateSmearerPlugin,
    "PixelTemplateSmearerPlugin"
);
