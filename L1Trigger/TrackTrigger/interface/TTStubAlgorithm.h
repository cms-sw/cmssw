/*! \class   TTStubAlgorithm
 *  \brief   Base class for any algorithm to be used
 *           in TTStubBuilder
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Andrew W. Rose
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

#ifndef L1_TRACK_TRIGGER_STUB_ALGO_BASE_H
#define L1_TRACK_TRIGGER_STUB_ALGO_BASE_H

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <sstream>
#include <string>
#include <map>
#include "classNameFinder.h"

template <typename T>
class TTStubAlgorithm {
protected:
  /// Data members
  const TrackerGeometry *const theTrackerGeom_;
  const TrackerTopology *const theTrackerTopo_;
  std::string className_;

public:
  /// Constructors
  TTStubAlgorithm(const TrackerGeometry *const theTrackerGeom,
                  const TrackerTopology *const theTrackerTopo,
                  std::string fName)
      : theTrackerGeom_(theTrackerGeom), theTrackerTopo_(theTrackerTopo) {
    className_ = classNameFinder<T>(fName);
  }

  /// Destructor
  virtual ~TTStubAlgorithm() {}

  /// Matching operations
  virtual void PatternHitCorrelation(
      bool &aConfirmation, int &aDisplacement, int &anOffset, float &anHardBend, const TTStub<T> &aTTStub) const {}
  // Removed real offset.  Ivan Reid 10/2019

  /// Algorithm name
  virtual std::string AlgorithmName() const { return className_; }

};  /// Close class

#endif
