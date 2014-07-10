#ifndef FastSimulation_Tracking_SeedingLayerSpec_h
#define FastSimulation_Tracking_SeedingLayerSpec_h

struct LayerSpec
{
    public:
        std::string name;
        unsigned int subDet;
        unsigned int side;
        unsigned int idLayer;
        
        LayerSpec():
            subDet(0),
            side(0),
            idLayer(0)
        {
        }

        inline bool operator==(const LayerSpec& layer) const
        {
            //return name == layer.name;
	        return (subDet==layer.subDet) && (side==layer.side) && (idLayer==layer.idLayer);
        }
        
        inline bool operator<(const LayerSpec& layer) const
        {
            //return name < layer.name;
	        return (subDet==layer.subDet) && (side==layer.side) && (idLayer==layer.idLayer);
        }

        inline std::string print() const
        {
            return name;
        }
};

#endif

