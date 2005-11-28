#ifndef GeomDetType_H
#define GeomDetType_H

#include <string>

class Topology;

class GeomDetType {
public:

  enum SubDetector {PixelBarrel, TIB, TOB, PixelEndcap, TID, TEC, 
     CSC, DT, RPCBarrel, RPCEndcap};

    GeomDetType( const std::string&, SubDetector);

    virtual ~GeomDetType();

    virtual const Topology& topology() const = 0;

    const std::string& name() const {return theName;}

    SubDetector subDetector() const {return theSubDet;}

    bool isTrackerStrip() const;
    bool isTrackerPixel() const;
    bool isTracker()      const;
    bool isRPC()          const;
    bool isMuon()         const;

private:

    std::string theName;
    SubDetector theSubDet;

};

#endif
