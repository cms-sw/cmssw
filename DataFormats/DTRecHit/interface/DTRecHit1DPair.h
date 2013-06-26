#ifndef DTRecHit_DTRecHit1DPair_H
#define DTRecHit_DTRecHit1DPair_H

/** \class DTRecHit1DPair
 *
 *  Composed recHit representing a pair of reconstructed hits
 *
 *  For each signal theLeftHit in the DT wire, two hits can be constructed, due to the
 *  Left/Right ambiguity, which can be solved only associating several hits
 *  together. This class describes the pair of points associated to a single
 *  TDC signal. The two hits can be accessed via recHits()
 *  method. The position is the average of the theLeftHit and theRightHit hits, namely the
 *  wire position.
 *
 *  $Date: 2006/04/05 16:43:55 $
 *  $Revision: 1.4 $
 *  \author S. Lacaprara & G. Cerminara
 */

#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include <utility>

class DTLayer;
class DTDigi;

class DTRecHit1DPair : public RecHit1D {
public:
  /// Constructor without components: must use setPos and Err!
  DTRecHit1DPair(const DTWireId& wireId,
		 const DTDigi& digi);

  /// Default constructor. Needed to write the RecHit into a STL container.
  DTRecHit1DPair();


  /// Destructor
  virtual ~DTRecHit1DPair();

  // Operations

  virtual DTRecHit1DPair * clone() const;


  /// Return the 3-dimensional local position.
  /// The average theLeftHit/theRightHit hits position, namely the wire position
  /// is returned. 
  virtual LocalPoint localPosition() const;


  /// Return the 3-dimensional error on the local position. 
  /// The error is defiened as half
  /// the distance between theLeftHit and theRightHit pos
  virtual LocalError localPositionError() const;


  /// Access to component RecHits.
  /// Return the two recHits (L/R)
  virtual std::vector<const TrackingRecHit*> recHits() const;


  /// Non-const access to component RecHits.
  /// Return the two recHits (L/R)
  virtual std::vector<TrackingRecHit*> recHits();


  /// Return the detId of the Det (a DTLayer).
  virtual DetId geographicalId() const;


  /// Return the digi time (ns) used to build the rechits
  float digiTime() const {
    return theLeftHit.digiTime();
  }


  /// Comparison operator, based on the wireId and the digi time
  bool operator==(const DTRecHit1DPair& hit) const;


  /// Inequality operator, defined as the mirror image of the comparions
  /// operator
  bool operator!=(const DTRecHit1DPair& hit) const {
    return !(*this==hit);
  }


  /// Return position in the local (layer) coordinate system for a
  /// certain hypothesis about the L/R cell side
  LocalPoint localPosition(DTEnums::DTCellSide lrside) const;


  /// Return position error in the local (layer) coordinate system for a
  /// certain hypothesis about the L/R cell side
  LocalError localPositionError(DTEnums::DTCellSide lrside) const;


  /// Set the 3-dimensional local position for the component hit
  /// corresponding to the given cell side. Default value is assumed for the error.
  void setPosition(DTEnums::DTCellSide lrside, const LocalPoint& point);


  /// Set the 3-dimensional local position and error for the component hit
  /// corresponding to the given cell side. Default value is assumed for the error.
  void setPositionAndError(DTEnums::DTCellSide lrside,
			   const LocalPoint& point, 
			   const LocalError& err);


  // Return the wireId
  DTWireId wireId() const {
    return theLeftHit.wireId();
  }


  /// Return the left/right DTRecHit1D
  const DTRecHit1D* componentRecHit(DTEnums::DTCellSide lrSide) const;

  
  /// Get the left and right 1D rechits (first and second respectively).
  std::pair<const DTRecHit1D*, const DTRecHit1D*> componentRecHits() const;


 private:

  /// Non const access to left/right DTRecHit1D
  DTRecHit1D* componentRecHit(DTEnums::DTCellSide lrSide);

  // The two rechits
  DTRecHit1D theLeftHit;
  DTRecHit1D theRightHit;

};


/// Ostream operator
std::ostream& operator<<(std::ostream& os, const DTRecHit1DPair& hit);

#endif
