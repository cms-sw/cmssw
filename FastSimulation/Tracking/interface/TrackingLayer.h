#ifndef FastSimulation_Tracking_TrackingLayer_h
#define FastSimulation_Tracking_TrackingLayer_h

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <sstream>

class TrackingLayer
{
    private:
        
    public:
        struct hashfct
        {
            inline size_t operator()(const TrackingLayer &layerSpec) const 
            {
                return layerSpec.getSubDetNumber()*10000+layerSpec.getLayerNumber()*100+layerSpec.getSideNumber()+1;
            }
        };
        struct eqfct
        {
            static hashfct gethash;
            inline bool operator()(const TrackingLayer &l1, const TrackingLayer &l2) const 
            {
                return l1.getSubDetNumber()==l2.getSubDetNumber() and l1.getLayerNumber()==l2.getLayerNumber() and l1.getSideNumber()==l2.getSideNumber();
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
        static eqfct _eqfct;
        static hashfct _hashfct;
        unsigned int _layerNumber;
        
        TrackingLayer():
            _subDet(Det::UNKNOWN),
            _side(Side::BARREL),
            _layerNumber(0)
        {
        }
        
        
        static TrackingLayer createFromDetId(
            const DetId& detId, 
            const TrackerTopology& trackerTopology
        )
        {
            TrackingLayer trackingLayer;
            uint32_t subdet=detId.subdetId();
            //BPix
            if ( subdet == PixelSubdetector::PixelBarrel )
            {
                trackingLayer._subDet=TrackingLayer::Det::PXB;
                trackingLayer._side=TrackingLayer::Side::BARREL;
                trackingLayer._layerNumber = trackerTopology.pxbLayer(detId);
            }
            //FPix
            else if ( subdet == PixelSubdetector::PixelEndcap )
            {
                trackingLayer._subDet=TrackingLayer::Det::PXD;
                if(trackerTopology.pxfSide(detId)==1)
                {
                    trackingLayer._side=TrackingLayer::Side::NEG_ENDCAP;
                }
                else if(trackerTopology.pxfSide(detId)==2)
                {
                    trackingLayer._side=TrackingLayer::Side::POS_ENDCAP;
                }
                else
                {
                    throw cms::Exception("FastSimulation/Tracking") <<"Tracker hit for seeding in FPix seems neither on positive nor on negative disk side: "<<trackerTopology.print(detId).c_str();
                }
                trackingLayer._layerNumber = trackerTopology.pxfDisk(detId);
            }
            //TIB
            else if ( subdet == StripSubdetector::TIB )
            {
                trackingLayer._subDet=TrackingLayer::Det::TIB;
                trackingLayer._side=TrackingLayer::Side::BARREL;
                trackingLayer._layerNumber = trackerTopology.tibLayer(detId);
            }
            //TID
            else if ( subdet == StripSubdetector::TID )
            {
                trackingLayer._subDet=TrackingLayer::Det::TID;
                if(trackerTopology.tidSide(detId)==1)
                {
                    trackingLayer._side=TrackingLayer::Side::NEG_ENDCAP;
                }
                else if(trackerTopology.tidSide(detId)==2)
                {
                    trackingLayer._side=TrackingLayer::Side::POS_ENDCAP;
                }
                else
                {
                    throw cms::Exception("FastSimulation/Tracking") <<"Tracker hit for seeding in TID seems neither on positive nor on negative disk side: "<<trackerTopology.print(detId).c_str();
                }
                trackingLayer._layerNumber = trackerTopology.tidWheel(detId);
            }
            //TOB
            else if ( subdet == StripSubdetector::TOB )
            {
                trackingLayer._subDet=TrackingLayer::Det::TOB;
                trackingLayer._side=TrackingLayer::Side::BARREL;
                trackingLayer._layerNumber = trackerTopology.tobLayer(detId);
            }
            //TEC
            else if ( subdet == StripSubdetector::TEC )
            {
                trackingLayer._subDet=TrackingLayer::Det::TEC;
                if(trackerTopology.tecSide(detId)==1)
                {
                    trackingLayer._side=TrackingLayer::Side::NEG_ENDCAP;
                }
                else if(trackerTopology.tecSide(detId)==2)
                {
                    trackingLayer._side=TrackingLayer::Side::POS_ENDCAP;
                }
                else
                {
                    throw cms::Exception("FastSimulation/Tracking") <<"Tracker hit for seeding in TEC seems neither on positive nor on negative disk side: "<<trackerTopology.print(detId).c_str();
                }
                trackingLayer._layerNumber = trackerTopology.tecWheel(detId);
            }
            else
            {
                throw cms::Exception("FastSimulation/Tracking") << "Cannot determine seeding layer from DetId:"<<trackerTopology.print(detId).c_str()<<std::endl;
            }    
            //std::cout<<"LayerSpec::createFromDetId: "<<trackerTopology.print(detId).c_str()<<", parsed="<<seedingLayer.print().c_str()<<std::endl;
            return trackingLayer;
        }
        
        static TrackingLayer createFromString(std::string layerSpecification)
        {
            TrackingLayer trackingLayer;
            if (layerSpecification.substr(0,4)=="BPix" ) 
            {
                trackingLayer._subDet=TrackingLayer::Det::PXB;
                trackingLayer._side=TrackingLayer::Side::BARREL;
                trackingLayer._layerNumber = std::atoi(layerSpecification.substr(4,1).c_str());
            }
            else if (layerSpecification.substr(0,4)=="FPix" ) 
            {
                trackingLayer._subDet=TrackingLayer::Det::PXD;
                trackingLayer._layerNumber = std::atoi(layerSpecification.substr(4,1).c_str());
                if(layerSpecification.substr(layerSpecification.size()-3)=="pos")
                {
                    trackingLayer._side = TrackingLayer::Side::POS_ENDCAP;
                }
                else if (layerSpecification.substr(layerSpecification.size()-3)=="neg")
                {
                    trackingLayer._side = TrackingLayer::Side::NEG_ENDCAP;
                }
                else
                {
                    throw cms::Exception("FastSimulation/Tracking/python")
                        <<"FPix seeding layer configuration '"<<layerSpecification.c_str()<<"' does not specify the side correctly!";
                }
                
            }
            else if (layerSpecification.substr(0,3)=="TIB")
            {
                trackingLayer._subDet=TrackingLayer::Det::TIB;
                trackingLayer._side=TrackingLayer::Side::BARREL;
                trackingLayer._layerNumber = std::atoi(layerSpecification.substr(3,1).c_str());
            }
            else if (layerSpecification.substr(0,4)=="MTIB")
            {
                trackingLayer._subDet=TrackingLayer::Det::TIB;
                trackingLayer._side=TrackingLayer::Side::BARREL;
                trackingLayer._layerNumber = std::atoi(layerSpecification.substr(4,1).c_str());
            }
       
            else if (layerSpecification.substr(0,3)=="TID") 
            {
                trackingLayer._subDet=TrackingLayer::Det::TID;
                trackingLayer._layerNumber = std::atoi(layerSpecification.substr(3,1).c_str());
                if (layerSpecification.substr(layerSpecification.size()-3)=="pos")
                {
                    trackingLayer._side = TrackingLayer::Side::POS_ENDCAP;
                }
                else if (layerSpecification.substr(layerSpecification.size()-3)=="neg")
                {
                    trackingLayer._side = TrackingLayer::Side::NEG_ENDCAP;
                }
                else
                {
                    throw cms::Exception("FastSimulation/Tracking/python")
                        <<"TID seeding layer configuration '"<<layerSpecification.c_str()<<"' does not specify the side correctly!";
                }
            }
            else if (layerSpecification.substr(0,4)=="MTID") 
            {
                trackingLayer._subDet=TrackingLayer::Det::TID;
                trackingLayer._layerNumber = std::atoi(layerSpecification.substr(4,1).c_str());
                if (layerSpecification.substr(layerSpecification.size()-3)=="pos")
                {
                    trackingLayer._side = TrackingLayer::Side::POS_ENDCAP;
                }
                else if (layerSpecification.substr(layerSpecification.size()-3)=="neg")
                {
                    trackingLayer._side = TrackingLayer::Side::NEG_ENDCAP;
                }
                else
                {
                    throw cms::Exception("FastSimulation/Tracking/python")
                        <<"MTID seeding layer configuration '"<<layerSpecification.c_str()<<"' does not specify the side correctly!";
                }
            }
            else if (layerSpecification.substr(0,3)=="TOB" ) 
            {
                trackingLayer._subDet=TrackingLayer::Det::TOB;
                trackingLayer._side=TrackingLayer::Side::BARREL;
                trackingLayer._layerNumber = std::atoi(layerSpecification.substr(3,1).c_str());
            }
            else if (layerSpecification.substr(0,3)=="TEC" ) 
            {
                trackingLayer._subDet=TrackingLayer::Det::TEC;
                trackingLayer._layerNumber = std::atoi(layerSpecification.substr(3,1).c_str());
                if (layerSpecification.substr(layerSpecification.size()-3)=="pos")
                {
                    trackingLayer._side = TrackingLayer::Side::POS_ENDCAP;
                }
                else if (layerSpecification.substr(layerSpecification.size()-3)=="neg")
                {
                    trackingLayer._side = TrackingLayer::Side::NEG_ENDCAP;
                }
                else
                {
                    throw cms::Exception("FastSimulation/Tracking/python")
                        <<"TEC seeding layer configuration '"<<layerSpecification.c_str()<<"' does not specify the side correctly!";
                }


            } 
            else if (layerSpecification.substr(0,4)=="MTEC" ) 
            {
                trackingLayer._subDet=TrackingLayer::Det::TEC;
                trackingLayer._layerNumber = std::atoi(layerSpecification.substr(4,1).c_str());
                if (layerSpecification.substr(layerSpecification.size()-3)=="pos")
                {
                    trackingLayer._side = TrackingLayer::Side::POS_ENDCAP;
                }
                else if (layerSpecification.substr(layerSpecification.size()-3)=="neg")
                {
                    trackingLayer._side = TrackingLayer::Side::NEG_ENDCAP;
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
            
            return trackingLayer;
        }
        
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
        
        inline std::string toString() const
        {
            std::stringstream ss;
            switch (_subDet)
            {
                case TrackingLayer::Det::UNKNOWN:
                    ss<<"unknown";
                    break;
                case TrackingLayer::Det::PXB:
                    ss<<" BPix";
                    break;
                case TrackingLayer::Det::PXD:
                    ss<<" FPix";
                    break;
                case TrackingLayer::Det::TIB:
                    ss<<" TIB";
                    break;
                case TrackingLayer::Det::TID:
                    ss<<" TID";
                    break;
                case TrackingLayer::Det::TOB:
                    ss<<" TOB";
                    break;
                case TrackingLayer::Det::TEC:
                    ss<<" TEC";
                    break;
            }
            ss << _layerNumber;
            switch (_side)
            {
                case TrackingLayer::Side::BARREL:
                    break;
                case Side:: NEG_ENDCAP:
                    ss<<"_neg";
                    break;
                case TrackingLayer::Side:: POS_ENDCAP:
                    ss<< "_pos";
                    break;
            }
            
            return std::move(ss.str());
        }
        
        inline std::string toIdString() const
        {
            std::stringstream ss;
            ss<<getSubDetNumber()<<":"<<getLayerNumber()<<":"<<getSideNumber();
            return std::move(ss.str());
        }
};


#endif

