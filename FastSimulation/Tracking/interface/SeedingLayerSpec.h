#ifndef FastSimulation_Tracking_SeedingLayerSpec_h
#define FastSimulation_Tracking_SeedingLayerSpec_h

#include <sstream>

class LayerSpec
{
    private:
        
    public:
        struct hashfct
        {
            inline size_t operator()(const LayerSpec &layerSpec) const 
            {
                return layerSpec.getSubDet()*10000+layerSpec.getLayerNumber()*100+layerSpec.getSide()+1;
            }
        };
        struct eqfct
        {
            static hashfct gethash;
            inline bool operator()(const LayerSpec &l1, const LayerSpec &l2) const 
            {
                return l1.getSubDet()==l2.getSubDet() and l1.getLayerNumber()==l2.getLayerNumber() and l1.getSide()==l2.getSide();
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
        enum class Side { BARREL, NEG_ENDCAP, POS_ENDCAP};

        
        Det subDet;
        Side side;
        static eqfct _eqfct;
        static hashfct _hashfct;
        unsigned int layerNumber;
        
        LayerSpec():
            subDet(Det::UNKNOWN),
            side(Side::BARREL),
            layerNumber(0)
        {
        }
        
        
        static LayerSpec createFromDetId(
            const DetId& detId, 
            const TrackerTopology& trackerTopology
        )
        {
            LayerSpec seedingLayer;
            uint32_t subdet=detId.subdetId();
            //BPix
            if ( subdet == PixelSubdetector::PixelBarrel )
            {
                seedingLayer.subDet=Det::PXB;
                seedingLayer.side=Side::BARREL;
                seedingLayer.layerNumber = trackerTopology.pxbLayer(detId);
            }
            //FPix
            else if ( subdet == PixelSubdetector::PixelEndcap )
            {
                seedingLayer.subDet=Det::PXD;
                if(trackerTopology.pxfSide(detId)==1)
                {
                    seedingLayer.side=Side::NEG_ENDCAP;
                }
                else if(trackerTopology.pxfSide(detId)==2)
                {
                    seedingLayer.side=Side::POS_ENDCAP;
                }
                else
                {
                    throw cms::Exception("FastSimulation/Tracking") <<"Tracker hit for seeding in FPix seems neither on positive nor on negative disk side: "<<trackerTopology.print(detId).c_str();
                }
                seedingLayer.layerNumber = trackerTopology.pxfDisk(detId);
            }
            //TIB
            else if ( subdet == StripSubdetector::TIB )
            {
                seedingLayer.subDet=Det::TIB;
                seedingLayer.side=Side::BARREL;
                seedingLayer.layerNumber = trackerTopology.tibLayer(detId);
            }
            //TID
            else if ( subdet == StripSubdetector::TID )
            {
                seedingLayer.subDet=Det::TID;
                if(trackerTopology.tidSide(detId)==1)
                {
                    seedingLayer.side=Side::NEG_ENDCAP;
                }
                else if(trackerTopology.tidSide(detId)==2)
                {
                    seedingLayer.side=Side::POS_ENDCAP;
                }
                else
                {
                    throw cms::Exception("FastSimulation/Tracking") <<"Tracker hit for seeding in TID seems neither on positive nor on negative disk side: "<<trackerTopology.print(detId).c_str();
                }
                seedingLayer.layerNumber = trackerTopology.tidWheel(detId);
            }
            //TOB
            else if ( subdet == StripSubdetector::TOB )
            {
                seedingLayer.subDet=Det::TOB;
                seedingLayer.side=Side::BARREL;
                seedingLayer.layerNumber = trackerTopology.tobLayer(detId);
            }
            //TEC
            else if ( subdet == StripSubdetector::TEC )
            {
                seedingLayer.subDet=Det::TEC;
                if(trackerTopology.tecSide(detId)==1)
                {
                    seedingLayer.side=Side::NEG_ENDCAP;
                }
                else if(trackerTopology.tecSide(detId)==2)
                {
                    seedingLayer.side=Side::POS_ENDCAP;
                }
                else
                {
                    throw cms::Exception("FastSimulation/Tracking") <<"Tracker hit for seeding in TEC seems neither on positive nor on negative disk side: "<<trackerTopology.print(detId).c_str();
                }
                seedingLayer.layerNumber = trackerTopology.tecWheel(detId);
            }
            else
            {
                throw cms::Exception("FastSimulation/Tracking") << "Cannot determine seeding layer from DetId:"<<trackerTopology.print(detId).c_str()<<std::endl;
            }    
            //std::cout<<"LayerSpec::createFromDetId: "<<trackerTopology.print(detId).c_str()<<", parsed="<<seedingLayer.print().c_str()<<std::endl;
            return seedingLayer;
        }
        
        static LayerSpec createFromString(std::string layerSpecification)
        {
            LayerSpec seedingLayer;
            if (layerSpecification.substr(0,4)=="BPix" ) 
            {
                seedingLayer.subDet=LayerSpec::Det::PXB;
                seedingLayer.side=LayerSpec::Side::BARREL;
                seedingLayer.layerNumber = std::atoi(layerSpecification.substr(4,1).c_str());
            }
            else if (layerSpecification.substr(0,4)=="FPix" ) 
            {
                seedingLayer.subDet=LayerSpec::Det::PXD;
                seedingLayer.layerNumber = std::atoi(layerSpecification.substr(4,1).c_str());
                if(layerSpecification.substr(layerSpecification.size()-3)=="pos")
                {
                    seedingLayer.side = LayerSpec::Side::POS_ENDCAP;
                }
                else if (layerSpecification.substr(layerSpecification.size()-3)=="neg")
                {
                    seedingLayer.side = LayerSpec::Side::NEG_ENDCAP;
                }
                else
                {
                    throw cms::Exception("FastSimulation/Tracking/python")
                        <<"FPix seeding layer configuration '"<<layerSpecification.c_str()<<"' does not specify the side correctly!";
                }
                
            }
            else if (layerSpecification.substr(0,3)=="TIB")
            {
                seedingLayer.subDet=LayerSpec::Det::TIB;
                seedingLayer.side=LayerSpec::Side::BARREL;
                seedingLayer.layerNumber = std::atoi(layerSpecification.substr(3,1).c_str());
            }
            else if (layerSpecification.substr(0,4)=="MTIB")
            {
                seedingLayer.subDet=LayerSpec::Det::TIB;
                seedingLayer.side=LayerSpec::Side::BARREL;
                seedingLayer.layerNumber = std::atoi(layerSpecification.substr(4,1).c_str());
            }
       
            else if (layerSpecification.substr(0,3)=="TID") 
            {
                seedingLayer.subDet=LayerSpec::Det::TID;
                seedingLayer.layerNumber = std::atoi(layerSpecification.substr(3,1).c_str());
                if (layerSpecification.substr(layerSpecification.size()-3)=="pos")
                {
                    seedingLayer.side = LayerSpec::Side::POS_ENDCAP;
                }
                else if (layerSpecification.substr(layerSpecification.size()-3)=="neg")
                {
                    seedingLayer.side = LayerSpec::Side::NEG_ENDCAP;
                }
                else
                {
                    throw cms::Exception("FastSimulation/Tracking/python")
                        <<"TID seeding layer configuration '"<<layerSpecification.c_str()<<"' does not specify the side correctly!";
                }
            }
            else if (layerSpecification.substr(0,4)=="MTID") 
            {
                seedingLayer.subDet=LayerSpec::Det::TID;
                seedingLayer.layerNumber = std::atoi(layerSpecification.substr(4,1).c_str());
                if (layerSpecification.substr(layerSpecification.size()-3)=="pos")
                {
                    seedingLayer.side = LayerSpec::Side::POS_ENDCAP;
                }
                else if (layerSpecification.substr(layerSpecification.size()-3)=="neg")
                {
                    seedingLayer.side = LayerSpec::Side::NEG_ENDCAP;
                }
                else
                {
                    throw cms::Exception("FastSimulation/Tracking/python")
                        <<"MTID seeding layer configuration '"<<layerSpecification.c_str()<<"' does not specify the side correctly!";
                }
            }
            else if (layerSpecification.substr(0,3)=="TOB" ) 
            {
                seedingLayer.subDet=LayerSpec::Det::TOB;
                seedingLayer.side=LayerSpec::Side::BARREL;
                seedingLayer.layerNumber = std::atoi(layerSpecification.substr(3,1).c_str());
            }
            else if (layerSpecification.substr(0,3)=="TEC" ) 
            {
                seedingLayer.subDet=LayerSpec::Det::TEC;
                seedingLayer.layerNumber = std::atoi(layerSpecification.substr(3,1).c_str());
                if (layerSpecification.substr(layerSpecification.size()-3)=="pos")
                {
                    seedingLayer.side = LayerSpec::Side::POS_ENDCAP;
                }
                else if (layerSpecification.substr(layerSpecification.size()-3)=="neg")
                {
                    seedingLayer.side = LayerSpec::Side::NEG_ENDCAP;
                }
                else
                {
                    throw cms::Exception("FastSimulation/Tracking/python")
                        <<"TEC seeding layer configuration '"<<layerSpecification.c_str()<<"' does not specify the side correctly!";
                }


            } 
            else if (layerSpecification.substr(0,4)=="MTEC" ) 
            {
                seedingLayer.subDet=LayerSpec::Det::TEC;
                seedingLayer.layerNumber = std::atoi(layerSpecification.substr(4,1).c_str());
                if (layerSpecification.substr(layerSpecification.size()-3)=="pos")
                {
                    seedingLayer.side = LayerSpec::Side::POS_ENDCAP;
                }
                else if (layerSpecification.substr(layerSpecification.size()-3)=="neg")
                {
                    seedingLayer.side = LayerSpec::Side::NEG_ENDCAP;
                }
                else
                {
                    throw cms::Exception("FastSimulation/Tracking/python")
                        <<"MTEC seeding layer configuration '"<<layerSpecification.c_str()<<"' does not specify the side correctly!";
                }
            } 
            else 
            { 
                throw cms::Exception("FastSimulation/Tracking/python")
                    << "Bad data naming in seeding layer configuration."
                    << "no case sensitive name of ['BPix','FPix','TIB','MTIB','TID','MTID','TOB','TEC','MTEC'] matches '"<<layerSpecification.c_str()<<"'";
            }
            //std::cout<<"LayerSpec::createFromString: "<<layerSpecification.c_str()<<", parsed="<<seedingLayer.print().c_str()<<std::endl;
            
            return seedingLayer;
        }
    
        inline unsigned int getSubDet() const
        {
            return static_cast<unsigned int>(subDet);
        }

        inline unsigned int getSide() const
        {
            return static_cast<unsigned int>(side);
        }
        
        inline unsigned int getLayerNumber() const
        {
            return layerNumber;
        }
        
        inline bool operator==(const LayerSpec& layer) const
        {
            return _eqfct(*this, layer);
        }
        
        inline bool operator<(const LayerSpec& layer) const
        {
            return _hashfct(*this)<_hashfct(layer);
        }
        
        inline std::string print() const
        {
            std::stringstream ss;
            switch (subDet)
            {
                case Det::UNKNOWN:
                    ss<<"unknown";
                    break;
                case Det::PXB:
                    ss<<" BPix";
                    break;
                case Det::PXD:
                    ss<<" FPix";
                    break;
                case Det::TIB:
                    ss<<" TIB";
                    break;
                case Det::TID:
                    ss<<" TID";
                    break;
                case Det::TOB:
                    ss<<" TOB";
                    break;
                case Det::TEC:
                    ss<<" TEC";
                    break;
            }
            ss << layerNumber;
            switch (side)
            {
                case Side::BARREL:
                    break;
                case Side:: NEG_ENDCAP:
                    ss<<"_neg";
                    break;
                case Side:: POS_ENDCAP:
                    ss<< "_pos";
                    break;
            }
            
            return ss.str();
        }
        
        inline std::string printN() const
        {
            std::stringstream ss;
            ss<<getSubDet()<<":"<<getLayerNumber()<<":"<<getSide();
            return ss.str();
        }
};


#endif

