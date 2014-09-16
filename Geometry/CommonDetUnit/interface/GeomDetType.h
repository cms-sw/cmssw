#ifndef GeomDetType_H
#define GeomDetType_H

#include <string>
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"


class Topology;

class GeomDetType {
public:
    typedef GeomDetEnumerators::SubDetector SubDetector;
  
    GeomDetType( const std::string& name, SubDetector subdet);

    virtual ~GeomDetType();

    virtual const Topology& topology() const = 0;

    const std::string& name() const {return theName;}

    SubDetector subDetector() const {return theSubDet;}

    bool isBarrel() const;
    bool isEndcap() const;

    bool isTrackerStrip() const;
    bool isTrackerPixel() const;
    bool isTracker()      const;
    bool isDT()           const;
    bool isCSC()          const;
    bool isRPC()          const;
    bool isGEM()          const;
    bool isMuon()         const;

private:

    std::string theName;
    SubDetector theSubDet;
};

#endif
