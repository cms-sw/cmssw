#ifndef Geometry_TrackerGeometryBuilder_RectangularPixelTopology_H
#define Geometry_TrackerGeometryBuilder_RectangularPixelTopology_H

  /**
   * Topology for rectangular pixel detector with BIG pixels.
   */
// Modified for the large pixels. Should work for barrel and forward.
// Danek Kotlinski & Michele Pioppi, 3/06.
// The bigger pixels are on the ROC boundries.
// For columns (Y direction, longer direction):
//  the normal pixel are 150um long, big pixels are 300um long, 
//  the pixel index goes from 0 to 416 (or less for smaller modules)
//  the big pixel are in 0, 52,104,156,208,260,312,363
//                      51,103,155,207,259,311,363,415 .
// For rows (X direction, shorter direction):
//  the normal pixel are 100um wide, big pixels are 200um wide, 
//  the pixel index goes from 0 to 159 (or less for smaller modules)
//  the big pixel are in 79,80.
// The ROC has rows=80, cols=52.
// There are a lot of hardwired constants, sorry but this is a very 
// specific class. For any other sensor design it has to be rewritten.

// G. Giurgiu 11/27/06 ---------------------------------------------
// Check whether the pixel is at the edge of the module by adding the 
// following functions (ixbin and iybin are the pixel row and column 
// inside the module):
// bool isItEdgePixelInX (int ixbin) 
// bool isItEdgePixelInY (int iybin) 
// bool isItEdgePixel (int ixbin, int iybin) 
// ------------------------------------------------------------------
// Add the individual measurement to local trasformations classes 01/07 d.k.
// ------------------------------------------------------------------
// Add big pixel flags for cluster range 15/3/07 V.Chiochia

#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"
#include <iostream>
#include <iomanip>

namespace {
  const float EPS = 0.001; // accuray in pixel units, so about 0.1 um
  const float EPSCM = 0.00001; // accuray in cm, so about 0.1 um
  const bool TP_DEBUG = false; // print flag
}

class RectangularPixelTopology : public PixelTopology {
 private:
  // This is temporary before we find a better way
  static const int ROWS_PER_ROC = 80;     // Num of cols per ROC
  static const int COLS_PER_ROC = 52;     // Num of Rows per ROC
  static const int BIG_PIX_PER_ROC_X = 1; // in x direction, rows
  static const int BIG_PIX_PER_ROC_Y = 2; // in y direction, cols

public:

  // Constructor, initilize 
  RectangularPixelTopology( int nrows, int ncols, float pitchx, 
			    float pitchy) :
    m_nrows(nrows), m_ncols(ncols), 
    m_pitchx(pitchx), m_pitchy(pitchy) {
				
    //using std::cout;
    //using std::endl;

    // Calculate the edge of the active sensor with respect to the center,
    // that is simply the half-size.       
    // Take into account large pixels
    m_xoffset = -(m_nrows + BIG_PIX_PER_ROC_X*m_nrows/ROWS_PER_ROC)/2. * 
      m_pitchx;
    m_yoffset = -(m_ncols + BIG_PIX_PER_ROC_Y*m_ncols/COLS_PER_ROC)/2. * 
      m_pitchy;

    if(TP_DEBUG) std::cout<<" RectangularPixelTopology: "
		  <<m_nrows<<" "<<m_ncols<<" "
		  <<m_pitchx<<" "<<m_pitchy<<" "<<m_xoffset<<" "<<m_yoffset
		  <<BIG_PIX_PER_ROC_X<<" "<<BIG_PIX_PER_ROC_Y<<" "
		  <<ROWS_PER_ROC<<" "<<COLS_PER_ROC<<std::endl;
  }

  // Topology interface, go from Masurement to Local corrdinates
  // pixel coordinates (mp) -> cm (LocalPoint)
  virtual LocalPoint localPosition( const MeasurementPoint& mp) const;

  // Transform LocalPoint to Measurement. Call pixel().
  virtual MeasurementPoint measurementPosition( const LocalPoint& lp) 
      const {
    std::pair<float,float> p = pixel(lp);
    return MeasurementPoint( p.first, p.second);
  }

  // PixelTopology interface. 
  // Transform LocalPoint in cm to measurement in pitch units.
  virtual std::pair<float,float> pixel( const LocalPoint& p) const;

  // Errors
  // Error in local (cm) from the masurement errors
  virtual LocalError localError( const MeasurementPoint&,
				 const MeasurementError& ) const;
  // Errors in pitch units from localpoint error (in cm)
  virtual MeasurementError measurementError( const LocalPoint&, 
					     const LocalError& ) const;
  
  //
  // Transform LocalPoint to channel. Call pixel()
  virtual int channel( const LocalPoint& lp) const {
    std::pair<float,float> p = pixel(lp);
    return PixelChannelIdentifier::pixelToChannel( int(p.first), 
						   int(p.second));
  }


  // Transform measurement to local coordinates individually in each dimension
  virtual float localX(const float mpX) const;
  virtual float localY(const float mpY) const;

  //-------------------------------------------------------------
  // Return the BIG pixel information for a given pixel
  //
  virtual bool isItBigPixelInX(const int ixbin) const {
    return ( (ixbin == 79) || (ixbin == 80));
  } 
  virtual bool isItBigPixelInY(const int iybin) const {
    int iybin0 = iybin%52;
    return ( (iybin0 == 0) || (iybin0 == 51));
  } 
  //-------------------------------------------------------------
  // Return BIG pixel flag in a given pixel range
  //
  bool containsBigPixelInX(const int& ixmin, const int& ixmax) const;
  bool containsBigPixelInY(const int& iymin, const int& iymax) const;


  // Check whether the pixel is at the edge of the module
  bool isItEdgePixelInX (int ixbin) const {
    return ( (ixbin == 0) || (ixbin == (m_nrows-1)) );
  } 
  bool isItEdgePixelInY (int iybin) const {
    return ( (iybin == 0) || (iybin == (m_ncols-1)) );
  } 
  bool isItEdgePixel (int ixbin, int iybin) const {
    return ( isItEdgePixelInX( ixbin ) || isItEdgePixelInY( iybin ) );
  } 

  //------------------------------------------------------------------
  // Return pitch
  virtual std::pair<float,float> pitch() const {
    return std::pair<float,float>( float(m_pitchx), float(m_pitchy));
  }
  // Return number of rows
  virtual int nrows() const {
    return m_nrows;
  }
  // Return number of cols
  virtual int ncolumns() const {
    return m_ncols;
  }
  
private:
  int m_nrows;
  int m_ncols;
  float m_pitchx;
  float m_pitchy;
  float m_xoffset;
  float m_yoffset;
};

#endif


