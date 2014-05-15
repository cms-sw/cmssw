/*! \brief   Implementation of methods of TTStubAlgorithm_window_WindowFinder classes
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Andrew W. Rose
 *  \date   2013, Jul 18
 *
 */

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_window_WindowFinder.h"
#include <iostream>

/// Constructor
WindowFinder::WindowFinder( const StackedTrackerGeometry *aGeometry,
                            double aPtScalingFactor,
                            double aIPwidth,
                            double aRowResolution,
                            double aColResolution )
  : mGeometry( aGeometry ),
    mPtScalingFactor( aPtScalingFactor ),
    mIPwidth( aIPwidth ),
    mRowResolution( aRowResolution ),
    mColResolution( aColResolution ),
    mMinrow(0), mMaxrow(0), mMincol(0), mMaxcol(0),
    mLastId(0), mlastInnerRow(-1), mlastInnerCol(-1){}

/// Destructor
WindowFinder::~WindowFinder(){}

/// Dump hit
void WindowFinder::dumphit( const StackedTrackerDetId & anId,
                            unsigned int hitIdentifier,
                            const double & aInnerRow,
                            const double & aInnerColumn )
{
  const PixelGeomDetUnit* detunit = reinterpret_cast< const PixelGeomDetUnit* > (mGeometry->idToDetUnit( anId, hitIdentifier ));

  /// Find the centre of the Pixel
  MeasurementPoint mp( aInnerRow + (0.5*mRowResolution), aInnerColumn + (0.5*mColResolution) );
  LocalPoint LP  = detunit->topology().localPosition( mp );
  GlobalPoint GP = detunit->surface().toGlobal( LP );
  std::cout << (hitIdentifier?"INNER":"OUTER") << " -> eta = " << GP.eta() << std::endl;
}

/// Get window
StackedTrackerWindow WindowFinder::getWindow( const StackedTrackerDetId & anId,
                                              const double & aInnerRow,
                                              const double & aInnerColumn )
{
  /// Particular cases
  if ( (anId == mLastId) && (mlastInnerRow == aInnerRow) && (mlastInnerCol == aInnerColumn) )
  {
    StackedTrackerWindow thisWindow = StackedTrackerWindow( mMinrow, mMaxrow, mMincol, mMaxcol );
    return thisWindow;
  }

  mlastInnerRow = aInnerRow;
  mlastInnerCol = aInnerColumn;

  if (anId != mLastId)
  {
    mLastId = anId;
    mInnerDet = const_cast< PixelGeomDetUnit* >(reinterpret_cast< const PixelGeomDetUnit* > (mGeometry->idToDetUnit( anId, 0 )));
    mOuterDet = const_cast< PixelGeomDetUnit* >(reinterpret_cast< const PixelGeomDetUnit* > (mGeometry->idToDetUnit( anId, 1 )));

    mHalfPixelLength = mInnerDet->specificTopology().pitch().second * mColResolution * 0.5;
    mSeparation = mInnerDet->surface().localZ( mOuterDet->position() );
    if ( mSeparation < 0 )
      mSeparation = -mSeparation;

    mInnerDetRadius  = mInnerDet->position().perp();
    mInnerDetPhi     = mInnerDet->position().phi();
  }

  /// Find the bounds of the inner "pixel" in pixel units
  /// Remember to use the centre of the "pixel"
  MeasurementPoint MP_INNER( aInnerRow + (0.5*mRowResolution), aInnerColumn + (0.5*mColResolution) );

  /// Find the bounds of the inner "pixel" in cm
  LocalPoint       LP_INNER = mInnerDet->topology().localPosition( MP_INNER );

  /// Find the positions of the inner "pixels" corners in global coordinates
  GlobalPoint      GP_INNER = mInnerDet->surface().toGlobal( LP_INNER );

  /// Calculate the maximum allowed track angle to the tangent at the "pixels" bounds
  double           PHI = asin( mPtScalingFactor * GP_INNER.perp() );

  /// Calculate the angle of the sensor to the tangent at the bounds
  double           PixelAngle = acos( sin( mInnerDetPhi - GP_INNER.phi() ) * mInnerDetRadius / LP_INNER.x() );

  /// Calculate the deviation in the r-phi direction
  double           deltaXminus = ( mSeparation * tan( PixelAngle - PHI ));  
  double           deltaXplus  = ( mSeparation * tan( PixelAngle + PHI ));  

  /// Inner pixel z-bounds
  double           PIXEL_Z_PLUS  = GP_INNER.z()+mHalfPixelLength;
  double           PIXEL_Z_MINUS = GP_INNER.z()-mHalfPixelLength;

  /// IP z-bounds
  double           IP_Z_PLUS  =  mIPwidth;
  double           IP_Z_MINUS = -mIPwidth;

  /// Stack radial separation through inner hit
  double                R_SEPARATION =  mSeparation / cos( PixelAngle );
  if (R_SEPARATION < 0)
    R_SEPARATION = -R_SEPARATION;

  /// Calculate the deviation in the z direction
  double           deltaZminus = (PIXEL_Z_MINUS-IP_Z_PLUS) * R_SEPARATION / GP_INNER.perp();  
  double           deltaZplus  = (PIXEL_Z_PLUS-IP_Z_MINUS) * R_SEPARATION / GP_INNER.perp();    

  /// Make boundary points in the inner reference frame
  LocalPoint LP_OUTER_PLUS( LP_INNER.x()-deltaXplus , LP_INNER.y()-mHalfPixelLength-deltaZplus , -mSeparation );
  LocalPoint LP_OUTER_MINUS( LP_INNER.x()-deltaXminus , LP_INNER.y()+mHalfPixelLength-deltaZminus , -mSeparation );

  /// Migrate into the global frame
  GlobalPoint GP_OUTER_PLUS = mInnerDet ->surface().toGlobal(LP_OUTER_PLUS);
  GlobalPoint GP_OUTER_MINUS = mInnerDet ->surface().toGlobal(LP_OUTER_MINUS);

  /// Migrate into the local frame of the outer det
  LocalPoint LP_OUTER_PLUS_2 = mOuterDet ->surface().toLocal(GP_OUTER_PLUS);
  LocalPoint LP_OUTER_MINUS_2 = mOuterDet ->surface().toLocal(GP_OUTER_MINUS);

  /// Convert into pixel units
  std::pair<float,float> PLUS = mOuterDet -> specificTopology().pixel(LP_OUTER_PLUS_2);
  std::pair<float,float> MINUS = mOuterDet -> specificTopology().pixel(LP_OUTER_MINUS_2);

  /// Calculate window coordinates
  mMinrow = mRowResolution * floor( PLUS.first / mRowResolution ); 
  mMincol = mColResolution * floor( PLUS.second / mColResolution );
  mMaxrow = mRowResolution * floor( MINUS.first / mRowResolution );
  mMaxcol = mColResolution * floor( MINUS.second / mColResolution ); 

  if (mMinrow>mMaxrow)
    std::swap(mMinrow,mMaxrow);
  if (mMincol>mMaxcol)
    std::swap(mMincol,mMaxcol);

  /// Return the window
  StackedTrackerWindow theWindow = StackedTrackerWindow( mMinrow, mMaxrow, mMincol, mMaxcol );
  return theWindow;
}

