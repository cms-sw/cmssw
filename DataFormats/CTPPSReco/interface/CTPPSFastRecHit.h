#ifndef CTPPSFastRecHit_H
#define CTPPSFastRecHit_H

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include <vector>

class CTPPSFastRecHit {
    public: 
        //destructor
        // ~CTPPSFastRecHit() {}
        CTPPSFastRecHit() : theDetUnitId(0) {}
        // constructor
        // requires the DetId, the hit position, the ToF and the CellId 
        // For the Tracker, ToF and CellId = 0
        // For the timing x = x_CellId, y = y_CellId, z = z_detector

        CTPPSFastRecHit(const Local3DPoint& entry, unsigned int detId, float tof, unsigned int cellId):
            theEntryPoint( entry), 
            theDetUnitId( detId),
            theTof(tof),       
            theCellId( cellId) {}
        /// Entry point in the local Det frame 
        Local3DPoint entryPoint() const {return theEntryPoint;}

        /* Time of flight in nanoseconds from the primary interaction
         *  to the entry point. Always positive in a PSimHit,
         *  but may become negative in a SimHit due to bunch assignment.
         */
        float timeOfFlight() const {return tof();}

        /// deprecated name for timeOfFlight()
        float tof() const {return theTof;}

        /* The DetUnit identifier, to be interpreted in the context of the
         *  detector system that produced the hit.
         *  For CTPPS its content has: Detector(CTPPS), SubDet (Tracker or Timing)
         *  ArmF(z>0)/ArmB(z<0),  Pot and Plane (= 0)
         */
        unsigned int detUnitId() const {return theDetUnitId;}
        //the ToF cell number
        unsigned int cellId() const {return theCellId;}

        void setTof(float tof) {theTof=tof;}
	
	void setLocal3DPoint(const Local3DPoint& entry){theEntryPoint = entry;}
	
	void setDetUnitId(unsigned int detId){theDetUnitId = detId;}

	void setCellId(unsigned int cellId){theCellId = cellId;}


		
    protected: 
        // properties
        Local3DPoint theEntryPoint; // position at entry
        // association
        unsigned int theDetUnitId;
        float theTof; // Time Of Flight 
        unsigned int theCellId;
}; 


#endif //CTPPSFastRecHit_H
