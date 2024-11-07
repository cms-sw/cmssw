#ifndef CLUSTER_H
#define CLUSTER_H

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

/**
 * @class Cluster
 * @brief Class to represent a single cluster in the phase 2 tracker
 */

class Cluster
{
    private:

        unsigned int z_;
        double x_;
        unsigned int width_;
        unsigned int chipId_;
        unsigned int sclusteraddress_;
        unsigned int mipbit_;
        unsigned int cicId_;
        TrackerGeometry::ModuleType ClusterType_;

    public:
    
        Cluster ( unsigned int& z,  double x,  unsigned int& width,  unsigned int& chipId,  unsigned int& sclusteraddress,  unsigned int& mipbit, unsigned int& cicid , TrackerGeometry::ModuleType& type) 
            : z_(z), x_(x), width_(width), chipId_(chipId), sclusteraddress_(sclusteraddress), mipbit_(mipbit), cicId_(cicid), ClusterType_(type) {}
        ~Cluster() {}

        TrackerGeometry::ModuleType getClusterType() { return ClusterType_; }

        unsigned int getZ() { return z_; }
        double getX() { return x_; }
        unsigned int getWidth() { return width_; }
        unsigned int getChipId() { return chipId_; }
        unsigned int getSclusterAddress() { return sclusteraddress_; }
        unsigned int getMipBit() { return mipbit_; }
        unsigned int getCicId() { return cicId_; }
};

#endif