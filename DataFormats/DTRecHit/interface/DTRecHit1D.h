#ifndef DTRecHit_DTRecHit1D_H
#define DTRecHit_DTRecHit1D_H

/** \class DTRecHit1D
 *
 *  1D RecHit for Muon Barrel DT 
 *  The main feature of muon Barrel RecHits is that they are created in pair,
 *  due to left/right ambiguity (the pair is described by \class
 *  DTRecHit1DPair). The coordiante measured is always the x (in Det frame)
 *
 *
 *  \author S. Lacaprara, G. Cerminara
 */

//#include "DataFormats/GeometrySurface/interface/LocalError.h"

#include "DataFormats/TrackingRecHit/interface/RecHit1D.h"

#include "DataFormats/DTRecHit/interface/DTEnums.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

class DTLayer;
class GeomDet;

class DTRecHit1D : public RecHit1D {
public:
  /// Constructor from wireId and digi time only.
  DTRecHit1D(const DTWireId& wireId, DTEnums::DTCellSide lr, float digiTime);

  /// Default constructor
  DTRecHit1D();

  /// Constructor from a local position, wireId and digi time.
  /// The 3-dimensional local error is defined as
  /// resolution (the cell resolution) for the coordinate being measured
  /// and 0 for the two other coordinates
  DTRecHit1D(const DTWireId& wireId, DTEnums::DTCellSide lr, float digiTime, const LocalPoint& pos);

  /// Constructor from a local position and error, wireId and digi time.
  DTRecHit1D(
      const DTWireId& wireId, DTEnums::DTCellSide lr, float digiTime, const LocalPoint& pos, const LocalError& err);

  /// Destructor
  ~DTRecHit1D() override;

  /// Return the 3-dimensional local position
  LocalPoint localPosition() const override { return theLocalPosition; }

  /// Return the 3-dimensional error on the local position
  LocalError localPositionError() const override { return theLocalError; }

  DTRecHit1D* clone() const override;

  /// No components rechits: it returns a null vector
  std::vector<const TrackingRecHit*> recHits() const override;

  /// No components rechits: it returns a null vector
  std::vector<TrackingRecHit*> recHits() override;

  /// The side of the cell
  DTEnums::DTCellSide lrSide() const { return theLRSide; }

  /// Set local position
  void setPosition(LocalPoint pos) { theLocalPosition = pos; }

  /// Set local position error
  void setError(LocalError err) { theLocalError = err; }

  /// Set the local position and its error
  void setPositionAndError(LocalPoint pos, LocalError err) {
    theLocalPosition = pos;
    theLocalError = err;
  }

  /// Return the wireId
  DTWireId wireId() const { return theWireId; }

  /// Return the time (ns) of the digi used to build the rechit
  float digiTime() const { return theDigiTime; }

  /// Comparison operator, based on the wireId and the digi time
  bool operator==(const DTRecHit1D& hit) const;

private:
  // The wire id
  DTWireId theWireId;

  // Left/Right side code
  DTEnums::DTCellSide theLRSide;

  // The digi time used to reconstruct the hit
  float theDigiTime;

  // Position and error in the Local Ref. Frame of the DTLayer
  LocalPoint theLocalPosition;
  LocalError theLocalError;
};
#endif

/// The ostream operator
std::ostream& operator<<(std::ostream& os, const DTRecHit1D& hit);
