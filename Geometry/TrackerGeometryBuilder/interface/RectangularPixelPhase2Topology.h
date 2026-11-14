#ifndef Geometry_TrackerGeometryBuilder_RectangularPixelPhase2Topology_H
#define Geometry_TrackerGeometryBuilder_RectangularPixelPhase2Topology_H

/**
   * Topology for rectangular pixel detector with BIG pixels.
   */
// 2023/11 - Re-written for phase-2 pixels.
// G.Bardelli (UniFI/INFN)
// F.Luongo (UniTO/INFN)
// E.Migliore (UniTO/INFN)

#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class RectangularPixelPhase2Topology final : public PixelTopology {
public:
  // Constructor, initialize
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
				 bool BIG_PIX_TWOEDGES_X,
				 bool BIG_PIX_TWOEDGES_Y,
                                 int ROCS_X,
                                 int ROCS_Y)
      : m_pitchx(pitchx),
        m_pitchy(pitchy),
        m_nrows(nrows),
        m_ncols(ncols),
        m_ROWS_PER_ROC(ROWS_PER_ROC),            // Num of Rows per ROC
        m_COLS_PER_ROC(COLS_PER_ROC),            // Num of Cols per ROC
        m_BIG_PIX_PER_ROC_X(BIG_PIX_PER_ROC_X),  //
        m_BIG_PIX_PER_ROC_Y(BIG_PIX_PER_ROC_Y),  //
        m_BIG_PIX_PITCH_X(BIG_PIX_PITCH_X),
        m_BIG_PIX_PITCH_Y(BIG_PIX_PITCH_Y),
        m_BIG_PIX_TWOEDGES_X(BIG_PIX_TWOEDGES_X),
        m_BIG_PIX_TWOEDGES_Y(BIG_PIX_TWOEDGES_Y),
        m_ROCS_X(ROCS_X),  //
        m_ROCS_Y(ROCS_Y)   //
  {
    // Calculate the edge of the active sensor with respect to the center,
    // that is simply the half-size.
    // Take into account large pixels
    // Inner Tracker: quad planar is 670*0.025 + 2*0.0875; dual planar is 336* 0.025; 3D is 336*0.025; old geom is 672*faxe pitch_x
   
    m_xoffset = m_ROCS_X*( (m_ROWS_PER_ROC-BIG_PIX_PER_ROC_X)*m_pitchx + BIG_PIX_PER_ROC_X*BIG_PIX_PITCH_X );
    m_xoffset *= -0.5;  

    m_yoffset = m_ROCS_Y*( (m_COLS_PER_ROC-BIG_PIX_PER_ROC_Y)*m_pitchy + BIG_PIX_PER_ROC_Y*BIG_PIX_PITCH_Y );
    m_yoffset *= -0.5;  

    if ( m_BIG_PIX_PER_ROC_X != 0 )
      m_BIG_PIX_LIST_X = MakeBigPixelList(m_BIG_PIX_PER_ROC_X, m_ROWS_PER_ROC, m_ROCS_X, m_BIG_PIX_TWOEDGES_X);
    if ( m_BIG_PIX_PER_ROC_Y != 0 )
      m_BIG_PIX_LIST_Y = MakeBigPixelList(m_BIG_PIX_PER_ROC_Y, m_COLS_PER_ROC, m_ROCS_Y, m_BIG_PIX_TWOEDGES_Y);

    LogDebug("RectangularPixelPhase2Topology")
        << "nrows " << m_nrows << ", ncols " << m_ncols << ", pitchx " << m_pitchx << ", pitchy " << m_pitchy
        << ", xoffset " << m_xoffset << ", yoffset " << m_yoffset << ", BIG_PIX_PER_ROC_X " << BIG_PIX_PER_ROC_X
        << ", BIG_PIX_PER_ROC_Y " << BIG_PIX_PER_ROC_Y << ", BIG_PIX_PITCH_X " << BIG_PIX_PITCH_X
        << ", BIG_PIX_PITCH_Y " << BIG_PIX_PITCH_Y << ", ROWS_PER_ROC " << ROWS_PER_ROC << ", COLS_PER_ROC "
        << COLS_PER_ROC << ", ROCS_X " << ROCS_X << ", ROCS_Y " << ROCS_Y << "\nNROWS " << m_ROWS_PER_ROC * m_ROCS_X
        << ", NCOL " << m_COLS_PER_ROC * m_ROCS_Y;

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
  //
  float ConvertPixelToPitch(const float local_pos, const int n_rocrowsorcols, const int nroc, const float sensoractive_half_size, const float pitch, const int big_pix_per_roc, const float big_pix_pitch, const bool big_pix_twoedges) const; 

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
  //
  float localCoordinate(const float mp_pos, const int n_rocrowsorcols, const int nroc, const float m_offset, const float pitch, const int big_pix_per_roc, const float big_pix_pitch, const bool big_pix_twoedges) const;

  //-------------------------------------------------------------
  // Return the BIG pixel information for a given pixel 
  //
  bool isItBigPixelInX(const int ixbin) const override {
    bool no_big_pixel = (m_BIG_PIX_PER_ROC_X == 0);
    if (!no_big_pixel)
      no_big_pixel = std::count(m_BIG_PIX_LIST_X.begin(), m_BIG_PIX_LIST_X.end(), ixbin) == 0;

    return !no_big_pixel;
  }

  bool isItBigPixelInY(const int iybin) const override {
    bool no_big_pixel = (m_BIG_PIX_PER_ROC_Y == 0);
    if (!no_big_pixel)
      no_big_pixel = std::count(m_BIG_PIX_LIST_Y.begin(), m_BIG_PIX_LIST_Y.end(), iybin) == 0;

    return !no_big_pixel;
  }

  float pixelFractionInX(const int ixbin) const override {
    bool no_big_pixel = (m_BIG_PIX_PER_ROC_X == 0);

    if (no_big_pixel) {
      return 1.0f;
    } else {
      if ( isItBigPixelInX(ixbin) ) {
        return float(m_BIG_PIX_PITCH_X / m_pitchx);
      } else {
        return 1.0f;
      }
    }
  }

  float pixelFractionInY(const int iybin) const override {
    bool no_big_pixel = (m_BIG_PIX_PER_ROC_Y == 0);
    
    if (no_big_pixel) {
      return 1.0f;
    } else {
      if ( isItBigPixelInY(iybin) ) {
        return float(m_BIG_PIX_PITCH_Y / m_pitchy);
      } else {
        return 1.0f;
      }
    }
  }

  //-------------------------------------------------------------
  // Return BIG pixel flag in a given pixel range (assuming they are always at the edge between two CROCs)
  //
  bool containsBigPixelInX(int ixmin, int ixmax) const override {
    return containsBigPixel(ixmin, ixmax, m_BIG_PIX_LIST_X);
  }

  bool containsBigPixelInY(int iymin, int iymax) const override {
    return containsBigPixel(iymin, iymax, m_BIG_PIX_LIST_Y);
  }

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
  bool twoedgesbigpixelX() const { return m_BIG_PIX_TWOEDGES_X; }
  bool twoedgesbigpixelY() const { return m_BIG_PIX_TWOEDGES_Y; }

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
  bool m_BIG_PIX_TWOEDGES_X;
  bool m_BIG_PIX_TWOEDGES_Y;
  int m_ROCS_X;
  int m_ROCS_Y;
  std::vector<int> m_BIG_PIX_LIST_X;
  std::vector<int> m_BIG_PIX_LIST_Y;
		
  bool containsBigPixel(int iMin, int iMax, std::vector<int> m_BIG_PIX_LIST) const {

    // The interval contains no big pixel when either of the following is met:
    // - there are no big pixels
    // - no BigPixel is encompssed between iMin and iMax
    bool noBigPixel = true;
    for (auto & bpe: m_BIG_PIX_LIST) {
      if ( iMin <= bpe && bpe <= iMax ) noBigPixel = false;
    }

    return !noBigPixel;
  }

  // 
  std::vector<int> MakeBigPixelList(const int m_BIG_PIX_PER_ROC, const int m_NN_PER_ROC, const int m_ROCS, const bool m_TWOEDGES){
    
    std::vector<int> m_BIG_PIX_LIST;

    // BigPixels at both edges
    if ( m_TWOEDGES ) {
      for (int iroc = 0; iroc < m_ROCS; iroc++){
	for (int ibp=0; ibp<m_BIG_PIX_PER_ROC/2; ibp++){
	  m_BIG_PIX_LIST.push_back(     iroc*m_NN_PER_ROC + ibp);
	  m_BIG_PIX_LIST.push_back( (iroc+1)*m_NN_PER_ROC - 1 - ibp);
	}
      }
    } else {
    //  BigPixels central - allowed only when exactly 2 ROCs along this coordinate
      if ( m_ROCS != 2 ) {
	edm::LogError("RectangularPixelPhase2Topology") << "ERROR BigPixels only at the center but number of ROCS is: " << m_ROCS;
	std::cout << "MSG-%E%: " << m_ROCS << std::endl;
      } else {
	for (int ibp=0; ibp<m_BIG_PIX_PER_ROC; ibp++){
	  m_BIG_PIX_LIST.push_back(m_NN_PER_ROC - 1 - ibp);
	  m_BIG_PIX_LIST.push_back(m_NN_PER_ROC + ibp);
	}
      }
    }

    std::sort(m_BIG_PIX_LIST.begin(), m_BIG_PIX_LIST.end());        

    return m_BIG_PIX_LIST;
  }

};

#endif
