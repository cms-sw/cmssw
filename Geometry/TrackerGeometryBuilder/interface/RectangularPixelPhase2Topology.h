#ifndef Geometry_TrackerGeometryBuilder_RectangularPixelPhase2Topology_H
#define Geometry_TrackerGeometryBuilder_RectangularPixelPhase2Topology_H

/**
   * Topology for rectangular pixel detector with BIG pixels.
   */
// Re-written for phase-2 pixels.
// E. Migliore INFN/Universita Torino 2023/11
// The bigger pixels are on the ROC boundries.
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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class RectangularPixelPhase2Topology final : public PixelTopology {
public:
  // Constructor, initilize
  RectangularPixelPhase2Topology(int nrows,
                           int ncols,
                           float pitchx,
                           float pitchy,
                           int ROWS_PER_ROC,       // Num of Rows per ROC
                           int COLS_PER_ROC,       // Num of Cols per ROC
                           int BIG_PIX_PER_ROC_X,  // in x direction, rows
                           int BIG_PIX_PER_ROC_Y,  // in y direction, cols
                           float BIG_PIX_PITCH_X,
                           float BIG_PIX_PITCH_Y,
                           int ROCS_X,
                           int ROCS_Y)
      : m_pitchx(pitchx),
        m_pitchy(pitchy),
        m_nrows(nrows),
        m_ncols(ncols),
        m_ROWS_PER_ROC(ROWS_PER_ROC),  // Num of Rows per ROC
        m_COLS_PER_ROC(COLS_PER_ROC),  // Num of Cols per ROC
        m_BIG_PIX_PER_ROC_X(BIG_PIX_PER_ROC_X),  //
        m_BIG_PIX_PER_ROC_Y(BIG_PIX_PER_ROC_Y),  // 
	m_BIG_PIX_PITCH_X(BIG_PIX_PITCH_X),
        m_BIG_PIX_PITCH_Y(BIG_PIX_PITCH_Y),
        m_ROCS_X(ROCS_X),              // 
        m_ROCS_Y(ROCS_Y)               // 
        {
    // Calculate the edge of the active sensor with respect to the center,
    // that is simply the half-size.
    // Take into account large pixels
    //m_xoffset = -((m_nrows/2 - 2)*m_pitchx + 2*m_BIG_PIX_PITCH_X); //gbardell proto implementation fixing the number of big pix to the Phase 2 Geom, below a less harcoded attempt
    //m_yoffset = -((m_ncols/2 - 1)*m_pitchy + m_BIG_PIX_PITCH_Y);
    m_xoffset = -((m_nrows/2 - m_BIG_PIX_PER_ROC_X)*m_pitchx + m_BIG_PIX_PER_ROC_X*m_BIG_PIX_PITCH_X);  //gbardell: assuming the big pixel pitch is well computed it gets always in the middle regardless the number of big pixel.  Quad is 670*0.025 + 2*0.0875; double is 336* 0.025; 3D is 336*0.025; old geom is 672*faxe x pitch
    m_yoffset = -((m_ncols/2 - m_BIG_PIX_PER_ROC_Y)*m_pitchy + m_BIG_PIX_PER_ROC_Y*m_BIG_PIX_PITCH_Y);

    std::cout << "RectangularPixelPhase2Topology"
      //   LogDebug("RectangularPixelPhase2Topology")
	      << "nrows " << m_nrows << ", ncols " << m_ncols << ", pitchx " << m_pitchx
                                         << ", pitchy " << m_pitchy << ", xoffset " << m_xoffset << ", yoffset "
                                         << m_yoffset << ", BIG_PIX_PER_ROC_X " << BIG_PIX_PER_ROC_X
                                         << ", BIG_PIX_PER_ROC_Y " << BIG_PIX_PER_ROC_Y << ", BIG_PIX_PITCH_X "
                                         << BIG_PIX_PITCH_X << ", BIG_PIX_PITCH_Y " << BIG_PIX_PITCH_Y << ", ROWS_PER_ROC "
                                         << ROWS_PER_ROC << ", COLS_PER_ROC " << COLS_PER_ROC << ", ROCS_X " << ROCS_X
                                         << ", ROCS_Y " << ROCS_Y << "\nNROWS " << m_ROWS_PER_ROC * m_ROCS_X
                                         << ", NCOL " << m_COLS_PER_ROC * m_ROCS_Y
	      << std::endl;
  }

  // Topology interface, go from Masurement to Local corrdinates
  // pixel coordinates (mp) -> cm (LocalPoint)
  LocalPoint localPosition(const MeasurementPoint& mp) const override;

  // Transform LocalPoint to Measurement. Call pixel().
  MeasurementPoint measurementPosition(const LocalPoint& lp) const override {
    std::pair<float, float> p = pixel(lp);
    return MeasurementPoint(p.first, p.second);
  }

  // PixelTopology interface.
  // Transform LocalPoint in cm to measurement in pitch units.
  std::pair<float, float> pixel(const LocalPoint& p) const override;

  // Errors
  // Error in local (cm) from the masurement errors
  LocalError localError(const MeasurementPoint&, const MeasurementError&) const override;
  // Errors in pitch units from localpoint error (in cm)
  MeasurementError measurementError(const LocalPoint&, const LocalError&) const override;

  //-------------------------------------------------------------
  // Transform LocalPoint to channel. Call pixel()
  //
  int channel(const LocalPoint& lp) const override {
    std::pair<float, float> p = pixel(lp);
    return PixelChannelIdentifier::pixelToChannel(int(p.first), int(p.second));
  }

  //-------------------------------------------------------------
  // Transform measurement to local coordinates individually in each dimension
  //
  float localX(const float mpX) const override;
  float localY(const float mpY) const override;

  //-------------------------------------------------------------
  // Return the BIG pixel information for a given pixel (assuming they are always at the edge between two CROCs)
  //
  bool isItBigPixelInX(const int ixbin) const override {
    bool no_big_pixel = (m_BIG_PIX_PER_ROC_X == 0);
    if ( !no_big_pixel )
      no_big_pixel = abs((ixbin-m_nrows/2) + 0.5) > m_BIG_PIX_PER_ROC_X;
  
    return !no_big_pixel;
  }

  bool isItBigPixelInY(const int iybin) const override {
    bool no_big_pixel = (m_BIG_PIX_PER_ROC_Y == 0);
    if ( !no_big_pixel )
      no_big_pixel = abs((iybin-m_ncols/2) + 0.5) > m_BIG_PIX_PER_ROC_Y;
  
    return !no_big_pixel;
  }

  //-------------------------------------------------------------
  // Return BIG pixel flag in a given pixel range (assuming they are always at the edge between two CROCs)
  //
  bool containsBigPixelInX(int ixmin, int ixmax) const override {
    bool no_big_pixel = (m_BIG_PIX_PER_ROC_X == 0);
    if ( !no_big_pixel )      
      no_big_pixel = (ixmin < std::clamp(ixmin, m_nrows/2-m_BIG_PIX_PER_ROC_X, m_nrows/2+m_BIG_PIX_PER_ROC_X-1)) | (ixmax > std::clamp(ixmax, m_nrows/2-m_BIG_PIX_PER_ROC_X, m_nrows/2+m_BIG_PIX_PER_ROC_X-1)); 
	
     return !no_big_pixel;
  }
  
  bool containsBigPixelInY(int iymin, int iymax) const override {
    bool no_big_pixel = (m_BIG_PIX_PER_ROC_Y == 0);
    if ( !no_big_pixel )      
      no_big_pixel = (iymin < std::clamp(iymin, m_ncols/2-m_BIG_PIX_PER_ROC_Y, m_ncols/2+m_BIG_PIX_PER_ROC_Y-1)) | (iymax > std::clamp(iymax, m_ncols/2-m_BIG_PIX_PER_ROC_Y, m_ncols/2+m_BIG_PIX_PER_ROC_Y-1)); 
	
    return !no_big_pixel;
  }

  // @EM this is a dummy implementation ... 
  // big pixels in X e Y vengono letti dalla topologia
  // bisogna trovare dove viene fatto il parsing del file XML e costruita la topologia per passare i valori di questi bool
  // fare anche per MTD ???
  bool bigpixelsX() const override { return false; }
  bool bigpixelsY() const override { return false; }

  //-------------------------------------------------------------
  // Check whether the pixel is at the edge of the module
  //
  bool isItEdgePixelInX(int ixbin) const override { return ((ixbin == 0) | (ixbin == (m_nrows - 1))); }
  bool isItEdgePixelInY(int iybin) const override { return ((iybin == 0) | (iybin == (m_ncols - 1))); }
  bool isItEdgePixel(int ixbin, int iybin) const override {
    return (isItEdgePixelInX(ixbin) || isItEdgePixelInY(iybin));
  }

  //------------------------------------------------------------------
  // Return pitch
  std::pair<float, float> pitch() const override { return std::pair<float, float>(float(m_pitchx), float(m_pitchy)); }
  // Return number of rows
  int nrows() const override { return (m_nrows); }
  // Return number of cols
  int ncolumns() const override { return (m_ncols); }
  // mlw Return number of ROCS Y
  int rocsY() const override { return m_ROCS_Y; }
  // mlw Return number of ROCS X
  int rocsX() const override { return m_ROCS_X; }
  // mlw Return number of rows per roc
  int rowsperroc() const override { return m_ROWS_PER_ROC; }
  // mlw Return number of cols per roc
  int colsperroc() const override { return m_COLS_PER_ROC; }
  int bigpixperrocX() const { return m_BIG_PIX_PER_ROC_X; }
  int bigpixperrocY() const { return m_BIG_PIX_PER_ROC_Y; }
  float xoffset() const { return m_xoffset; }
  float yoffset() const { return m_yoffset; }
  float pitchbigpixelX() const { return m_BIG_PIX_PITCH_X; }
  float pitchbigpixelY() const { return m_BIG_PIX_PITCH_Y; }


private:
  float m_pitchx;
  float m_pitchy;
  float m_xoffset;
  float m_yoffset;
  int m_nrows;
  int m_ncols;
  int m_ROWS_PER_ROC;
  int m_COLS_PER_ROC;
  int m_BIG_PIX_PER_ROC_X;
  int m_BIG_PIX_PER_ROC_Y;
  float m_BIG_PIX_PITCH_X;
  float m_BIG_PIX_PITCH_Y;
  int m_ROCS_X;
  int m_ROCS_Y;
};

#endif
