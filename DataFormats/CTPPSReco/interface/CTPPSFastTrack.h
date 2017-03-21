#ifndef CTPPSFastTrack_H
#define CTPPSFastTrack_H

#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"

#include <vector>
class CTPPSFastTrack {
    public: 
        typedef math::XYZVector Vector;
        typedef math::XYZPoint Point;
        // ~CTPPSFastTrack() {}
        CTPPSFastTrack() : thet(0.),thexi(0.),theCellId(0),theTof(0.),theX1(0.),theY1(0.),theX2(0.),theY2(0.),momentum_(0, 0, 0),vertex_(0, 0, 0) {}
        // constructor
        CTPPSFastTrack(float t,float xi,unsigned int cellid ,float tof,float X1,float Y1,float X2,float Y2,const Vector &momentum,const Point &vertex): 
            thet(t),
            thexi(xi),
            theCellId(cellid),
            theTof(tof),
            theX1(X1),
            theY1(Y1),
            theX2(X2),
            theY2(Y2),
            momentum_(momentum),
            vertex_(vertex) {}	

        ////////////////////////////
        //
        /// track momentum vector
        const Vector &momentum() const;
        /// Reference point on the track
        const Point &referencePoint() const;
        // reference point on the track. This method is DEPRECATED, please use referencePoint() instead
        const Point &vertex() const ;
        /* Time of flight in nanoseconds from the primary interaction
         *  to the entry point. Always positive in a PSimHit,
         *  but may become negative in a SimHit due to bunch assignment.
         */
        float timeOfFlight() const {return tof();}

        float t() const {return thet;}

        float xi() const {return thexi;}

        float tof() const {return theTof;}
        
        float X1() const {return theX1;}

        float Y1() const {return theY1;}

        float X2() const {return theX2;}

        float Y2() const {return theY2;}
        float PX() const {return momentum_.x();}
        float PY() const {return momentum_.Y();}
        float PZ() const {return momentum_.Z();}
        float X0() const {return vertex_.x();}
        float Y0() const {return vertex_.Y();}
        float Z0() const {return vertex_.Z();}

        unsigned int cellId() const {return theCellId;}

        void setP(const Vector& momentum ) { momentum_ = momentum; }

        void setVertex(const Point &vertex) {vertex_ = vertex;}

        void setTof(float tof) {theTof=tof;}

        void setT(float t){thet = t;}

        void setXi(float xi){thexi = xi;}

        void setX1(float X1){theX1 = X1;}

        void setY1(float Y1){theY1 = Y1;}

        void setX2(float X2){theX2 = X2;}

        void setY2(float Y2){theY2 = Y2;}

        void setCellId(unsigned int cellid ){theCellId=cellid;}

    private: 
        float    thet;
        float    thexi;
        unsigned int   theCellId;
        float    theTof;
        float    theX1;
        float    theY1;
        float    theX2;
        float    theY2;
        Vector momentum_;
        Point vertex_;

}; 

#endif //CTPPSFastTrack_H
