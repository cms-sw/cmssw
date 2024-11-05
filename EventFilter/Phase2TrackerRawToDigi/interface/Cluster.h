#ifndef CLASS_H
#define CLASS_H

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

class Cluster
{
    private:

        unsigned int z;
        unsigned int x;
        unsigned int width;
        unsigned int chipId;
        unsigned int sclusteraddress;
        unsigned int mipbit;
        unsigned int cicId;
        TrackerGeometry::ModuleType ClusterType;

    public:
    
        Cluster ( unsigned int& z,  unsigned int& x,  unsigned int& width,  unsigned int& chipId,  unsigned int& sclusteraddress,  unsigned int& mipbit, unsigned int& cicid , TrackerGeometry::ModuleType& type) : z(z), x(x), width(width), chipId(chipId), sclusteraddress(sclusteraddress), mipbit(mipbit), cicId(cicid), ClusterType(type) {}
        ~Cluster() {}

        TrackerGeometry::ModuleType getClusterType() { return ClusterType; }

        unsigned int getZ() { return z; }
        unsigned int getX() { return x; }
        unsigned int getWidth() { return width; }
        unsigned int getChipId() { return chipId; }
        unsigned int getSclusterAddress() { return sclusteraddress; }
        unsigned int getMipBit() { return mipbit; }
        unsigned int getCicId() { return cicId; }
        

};

#endif