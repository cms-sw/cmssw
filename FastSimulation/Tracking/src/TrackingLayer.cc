#include "FastSimulation/Tracking/interface/TrackingLayer.h"

const TrackingLayer::eqfct TrackingLayer::_eqfct;
const TrackingLayer::hashfct TrackingLayer::_hashfct;

TrackingLayer::TrackingLayer():
    _subDet(Det::UNKNOWN),
    _side(Side::BARREL),
    _layerNumber(0)
{
}


TrackingLayer TrackingLayer::createFromDetId(
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

TrackingLayer TrackingLayer::createFromString(std::string layerSpecification)
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

std::string TrackingLayer::toString() const
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

std::string TrackingLayer::toIdString() const
{
    std::stringstream ss;
    ss<<getSubDetNumber()<<":"<<getLayerNumber()<<":"<<getSideNumber();
    return std::move(ss.str());
}

