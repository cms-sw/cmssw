#ifndef Geometry_CommonTopologies_DTTopology_H
#define Geometry_CommonTopologies_DTTopology_H

/** \class DTTopology
 *
 * interface for the DriftTube detector. 
 * Extends the Topology interface with methods relevant for
 * the DT detectors.
 *  
 *  $Date: 2005/11/20 11:43:01 $
 *  $Revision: 1.1 $
 *
 * \author R. Bellan - INFN Torino
 *
 */

/*
#include "Geometry/CommonDetAlgo/interface/LocalError.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/CommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonDetAlgo/interface/MeasurementError.h"
*/

#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/Vector/interface/LocalPoint.h"

class DTTopology: public Topology {
 public:
  
  //Constructor: number of wire in the layer and its lenght
  DTTopology(int nChannels,float lenght);

  virtual ~DTTopology() {}
  
  // Conversion between measurement coordinates
  // and local cartesian coordinates

  LocalPoint localPosition( const MeasurementPoint& ) const;

  LocalError localError( const MeasurementPoint&, const MeasurementError& ) const;

  MeasurementPoint measurementPosition( const LocalPoint&) const;

  MeasurementError measurementError( const LocalPoint&, const LocalError& ) const;

  // return the wire number, starting from a LocalPoint
  int channel( const LocalPoint& p) const;

  // return the x-wire position in the layer, starting from its wire number.
  float wirePosition(int wireNumber);
  
  // They return the cell dimensions:
  float cellWidth(){return theWidth;}
  float cellHeight(){return theHeight;}
  float cellLenght(){return theLength;}

private:
  float theNChannels;
  float theWidth;
  float theHeight;
  float theLength;

  Local2DPoint theOffSet;
};

#endif
