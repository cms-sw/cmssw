#ifndef Geometry_CommonTopologies_DTTopology_H
#define Geometry_CommonTopologies_DTTopology_H

/** \class DTTopology
 *
 * interface for the DriftTube detector. 
 * Extends the Topology interface with methods relevant for
 * the DT detectors.
 *  
 *  $Date: 2005/11/21 13:14:53 $
 *  $Revision: 1.3 $
 *
 * \author R. Bellan - INFN Torino
 *
 */

#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/Vector/interface/LocalPoint.h"

class DTTopology: public Topology {
 public:
  
  //Constructor: number of first wire, total # of wires in the layer and their lenght
  DTTopology(int firstWire=0, int nChannels=0,float lenght=0); // togliere =0 prima di commitare

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
  float wirePosition(int wireNumber) const;
  
  // They return the cell dimensions:
  const float cellWidth() const {return theWidth;}
  const float cellHeight() const {return theHeight;}
  const float cellLenght() const {return theLength;}
  
  // They return the dimensions of the sensible volume of the cell:
  const float sensibleWidth() const;
  const float sensibleHeight() const;
  
private: 
  int theFirstChannel;
  int theNChannels;
  
  static const float theWidth;
  static const float theHeight;
  float theLength;

  Local2DPoint theOffSet;
};

#endif
