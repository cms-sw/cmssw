#ifndef DTRecHit_DTRecHit1DPair_H
#define DTRecHit_DTRecHit1DPair_H

/** \class DTRecHit1DPair
 *
 *  Composed recHit representing a pair of reconstructed hits
 *
 *  For each signal theLeftHit in the DT wire, two hits can be constructed, due to the
 *  Left/Right ambiguity, which can be solved only associating several hits
 *  together. This class describes the pair of points associated to a single
 *  TDC signal. The two hits can be accessed via recHits() and getComponents()
 *  methods. The position is the average of the theLeftHit and theRightHit hits, namely the
 *  wire position. If the pair is used by a segment, then the ambiguity is
 *  solved, and so the position is that of the used hit and the recHit is
 *  declared "matched".
 *
 *  $Date: $
 *  $Revision: $
 *  \author S. Lacaprara & G. Cerminara
 */

#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
#include "DataFormats/DTDigi/interface/DTDigi.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"



class DTLayer;

class DTRecHit1DPair : public RecHit1D {
public:
  /// Constructor without components: must use setPos and Err!
  DTRecHit1DPair(const DTWireId& wireId,
		 const DTDigi& digi);


  /// Destructor
  virtual ~DTRecHit1DPair();

  // Operations

  virtual DTRecHit1DPair * clone() const;


  /// Return the 3-dimensional local position.
  /// If the hit is not used by a
  /// segment, the average theLeftHit/theRightHit hits position, namely the wire position
  /// is returned. If it's used, then the used component's position is
  /// returned.
  virtual LocalPoint localPosition() const;


  /// Return the 3-dimensional error on the local position. 
  /// If the hit is not matched, the error is defiened as half
  /// the distance between theLeftHit and theRightHit pos: is
  /// matched, the correct hit error is returned.
  virtual LocalError localPositionError() const;


  /// Access to component RecHits.
  /// Return the two recHits (L/R): if the L/R is set, return the appropraite
  /// recHit
  virtual std::vector<const TrackingRecHit*> recHits() const;


  /// Non-const access to component RecHits.
  /// Return the two recHits (L/R): if the L/R is set, return the appropraite
  /// recHit
  virtual std::vector<TrackingRecHit*> recHits();

 // FIXME: Remove dependencies from Geometry
//   /// Access to the GeomDet (the layer) (essentially the Surface, with alignment interface)
//   virtual const GeomDet& det() const {
//     return (*theDet);
//   }


  /// Return the detId of the Det (a DTLayer).
  virtual DetId geographicalId() const;


  /// true if the code for L/R cell side is set, namely if used in a segment 
  virtual bool isMatched() const ;


  /// Return the digi
  const DTDigi& digi() const {
    return theDigi;
  }


  /// Comparison operator, based on the wireId and the digi
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


  /// Set the L/R side once the hit are matched in a segment
  void setLRCode(DTEnums::DTCellSide lrside);


  /// Return the cell side (L/R/undefined)
  DTEnums::DTCellSide lrSide() const {
    return theLRSide;
  }


  // Return the wireId
  DTWireId wireId() const {
    return theLeftHit.wireId();
  }


 private:
  // Return the left/right DTRecHit1D
  const DTRecHit1D* recHit(DTEnums::DTCellSide lrSide) const;
  
  // Non const access to left/right DTRecHit1D
  DTRecHit1D* recHit(DTEnums::DTCellSide lrSide);

  // The digi
  DTDigi theDigi;  //FIXME: is it really needed      
  
 // FIXME: Remove dependencies from Geometry
//   // The layer
//   const GeomDet* theDet;

  // The side of the cell. It can be left/right/unknomw
  DTEnums::DTCellSide theLRSide;


  // The two rechits
  DTRecHit1D theLeftHit; //FIXME: Should it be a pointer??
  DTRecHit1D theRightHit; //FIXME: Should it be a pointer??

};


/// Ostream operator
std::ostream& operator<<(std::ostream& os, const DTRecHit1DPair& hit);

#endif
