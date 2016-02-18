#ifndef FastSimulation_Tracking_TrackingLayer_h
#define FastSimulation_Tracking_TrackingLayer_h

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <sstream>

class TrackingLayer
{
    public:
	struct hashfct
	{
	    hashfct() { };

	    inline size_t operator()(const TrackingLayer &layerSpec) const
	    {
		return (layerSpec.getSubDetNumber() * 10000
			+ layerSpec.getLayerNumber() * 100
			+ layerSpec.getSideNumber() + 1);
	    }
	};

	struct eqfct
	{
	    eqfct() { };

	    inline bool operator()(const TrackingLayer &l1, const TrackingLayer &l2) const
	    {
		return (l1.getSubDetNumber() == l2.getSubDetNumber()
			&& l1.getLayerNumber() == l2.getLayerNumber()
			&& l1.getSideNumber() == l2.getSideNumber());
	    }
	};
        
        enum class Det { 
            UNKNOWN, 
            PXB, //pixel barrel
            PXD, //pixel disk
            TIB, //tracker inner barrel
            TID, //tracker inner disk
            TOB, //tracker outer barrel
            TEC  //tracker endcap
        };
        enum class Side { 
            BARREL, NEG_ENDCAP, POS_ENDCAP
        };

        
        Det _subDet;
        Side _side;
        static const eqfct _eqfct;
        static const hashfct _hashfct;
        unsigned int _layerNumber;
        
        TrackingLayer();
        
        static TrackingLayer createFromDetId(
            const DetId& detId, 
            const TrackerTopology& trackerTopology
        );
        
        static TrackingLayer createFromString(std::string layerSpecification);
        
        inline TrackingLayer::Det getSubDet() const
        {
            return _subDet;
        }

        inline TrackingLayer::Side getSide() const
        {
            return _side;
        }
    
        inline unsigned int getSubDetNumber() const
        {
            return static_cast<unsigned int>(_subDet);
        }

        inline unsigned int getSideNumber() const
        {
            return static_cast<unsigned int>(_side);
        }
        
        inline unsigned int getLayerNumber() const
        {
            return _layerNumber;
        }
        
        inline bool operator==(const TrackingLayer& layer) const
        {
            return _eqfct(*this, layer);
        }
        
        inline bool operator<(const TrackingLayer& layer) const
        {
            return _hashfct(*this)<_hashfct(layer);
        }
        
        std::string toString() const;
        std::string toIdString() const;
};


#endif

