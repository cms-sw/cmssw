#ifndef FASTSIM_SIMPLIFIEDGEOMETRYFACTORY
#define FASTSIM_SIMPLIFIEDGEOMETRYFACTORY

class GeometricSearchTracker;
class MagneticField;
class DetLayer;
class BarrelDetLayer;
class ForwardDetLayer;

#include "memory"
#include "map"
#include "vector"
#include "string"

namespace edm
{
    class ParameterSet;
}

namespace fastsim
{
    class SimplifiedGeometry;
    class BarrelSimplifiedGeometry;
    class ForwardSimplifiedGeometry;
    class InteractionModel;
    class SimplifiedGeometryFactory
    {
    public:

	SimplifiedGeometryFactory(const GeometricSearchTracker * geometricSearchTracker,
		     const MagneticField & magneticField,
		     const std::map<std::string,fastsim::InteractionModel *> & interactionModelMap,
		     double magneticFieldHistMaxR,
		     double magneticFieldHistMaxZ);
	
	enum LayerType {BARREL,POSFWD,NEGFWD};

	std::unique_ptr<SimplifiedGeometry> createSimplifiedGeometry(LayerType type,
					   const edm::ParameterSet & cfg) const;

	std::unique_ptr<ForwardSimplifiedGeometry> createForwardSimplifiedGeometry(LayerType type,
							 const edm::ParameterSet & cfg) const;

	std::unique_ptr<BarrelSimplifiedGeometry> createBarrelSimplifiedGeometry(const edm::ParameterSet & cfg) const;
	
    private:
	const DetLayer * getDetLayer(const std::string & detLayerName,const GeometricSearchTracker & geometricSearchTracker) const;
	const GeometricSearchTracker * const geometricSearchTracker_;
	const MagneticField * const magneticField_;
	const std::map<std::string,fastsim::InteractionModel *> * interactionModelMap_;
	const double magneticFieldHistMaxR_;
	const double magneticFieldHistMaxZ_;
	std::map<std::string,const std::vector<BarrelDetLayer const *> *> barrelDetLayersMap_;
	std::map<std::string,const std::vector<ForwardDetLayer const *> *> forwardDetLayersMap_;
    };
}

#endif
