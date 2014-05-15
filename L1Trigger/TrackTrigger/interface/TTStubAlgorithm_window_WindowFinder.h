#ifndef L1_TRACK_TRIGGER_STUB_ALGO_WINDOW_AUX_H
#define L1_TRACK_TRIGGER_STUB_ALGO_WINDOW_AUX_H

#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include <memory>

/*! \class   StackedTrackerWindow
 *
 *  \author Andrew W. Rose
 *  \date   2013, Jul 18
 *
 */

class StackedTrackerWindow
{
  public:
    /// Default constructor
    StackedTrackerWindow(){}

    /// Another constructor
    StackedTrackerWindow( double aMinrow, double aMaxrow, double aMincol, double aMaxcol )
      : mMinrow(aMinrow),
        mMaxrow(aMaxrow),
        mMincol(aMincol),
        mMaxcol(aMaxcol){}

    /// Destructor
    virtual ~StackedTrackerWindow(){}

    /// Data members
    double mMinrow;
    double mMaxrow;
    double mMincol;
    double mMaxcol;
};

/*! \class   WindowFinder
 *
 *  \author Andrew W. Rose
 *  \date   2013, Jul 18
 *
 */

class WindowFinder
{
  public:
    /// Constructor
    explicit WindowFinder( const StackedTrackerGeometry *aGeometry,
                           double aPtScalingFactor,
                           double aIPwidth,
                           double aRowResolution,
                           double aColResolution );

    /// Destructor
    virtual ~WindowFinder();

    /// Other methods
    void dumphit( const StackedTrackerDetId & anId,
                  unsigned int hitIdentifier,
                  const double & aInnerRow,
                  const double & aInnerColumn );

    StackedTrackerWindow getWindow( const StackedTrackerDetId & anId,
                                    const double & aInnerRow,
                                    const double & aInnerColumn );
  
  private:
    /// Data members
    const StackedTrackerGeometry *mGeometry;
    double mPtScalingFactor;
    double mIPwidth;
    double mRowResolution;
    double mColResolution;

    /// These are the variables which need to be filled!
    double mMinrow, mMaxrow, mMincol, mMaxcol;

    /// As all hits in the same stack are tested sequentially,
    /// then cache the sensor parameters for speed!
    StackedTrackerDetId mLastId;
    PixelGeomDetUnit *mInnerDet, *mOuterDet;
    double mSeparation;
    double mHalfPixelLength;
    double mInnerDetRadius, mInnerDetPhi;
    double mlastInnerRow, mlastInnerCol;
};

#endif

