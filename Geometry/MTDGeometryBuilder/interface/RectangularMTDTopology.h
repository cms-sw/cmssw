#ifndef Geometry_MTDGeometryBuilder_RectangularMTDTopology_H
#define Geometry_MTDGeometryBuilder_RectangularMTDTopology_H

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
#include "DataFormats/ForwardDetId/interface/MTDChannelIdentifier.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class RectangularMTDTopology final : public PixelTopology {
public:
  // Constructor, initilize
  RectangularMTDTopology(int nrows,
                         int ncols,
                         float pitchx,
                         float pitchy,
                         int ROWS_PER_ROC,  // Num of Rows per ROC
                         int COLS_PER_ROC,  // Num of Cols per ROC
                         int ROCS_X,
                         int ROCS_Y,
                         float GAPxInterpad,  // Value given in cm
                         float GAPxBorder,    // Value given in cm
                         float GAPyInterpad,  // Value given in cm
                         float GAPyBorder)    // Value given in cm
      : m_pitchx(pitchx),
        m_pitchy(pitchy),
        m_nrows(nrows),
        m_ncols(ncols),
        m_ROWS_PER_ROC(ROWS_PER_ROC),  // Num of Rows per ROC
        m_COLS_PER_ROC(COLS_PER_ROC),  // Num of Cols per ROC
        m_ROCS_X(ROCS_X),              // 2 for SLHC
        m_ROCS_Y(ROCS_Y),              // 8 for SLHC
        m_GAPxInterpad(GAPxInterpad),
        m_GAPxBorder(GAPxBorder),
        m_GAPyInterpad(GAPyInterpad),
        m_GAPyBorder(GAPyBorder) {
    m_xoffset = -(m_nrows / 2.) * m_pitchx;
    m_yoffset = -(m_ncols / 2.) * m_pitchy;
    m_GAPxInterpadFrac = m_GAPxInterpad / m_pitchx;
    m_GAPxBorderFrac = m_GAPxBorder / m_pitchx;
    m_GAPyInterpadFrac = m_GAPyInterpad / m_pitchy;
    m_GAPyBorderFrac = m_GAPyBorder / m_pitchy;
  }

  // Topology interface, go from Masurement to Local module corrdinates
  // pixel coordinates (mp) -> cm (LocalPoint)
  LocalPoint localPosition(const MeasurementPoint& mp) const override;

  // Transform LocalPoint to Measurement. Call pixel().
  MeasurementPoint measurementPosition(const LocalPoint& lp) const override {
    std::pair<float, float> p = pixel(lp);
    return MeasurementPoint(p.first, p.second);
  }

  // PixelTopology interface.
  std::pair<float, float> pixel(const LocalPoint& p) const override;

  //check whether LocalPoint is inside the pixel active area
  bool isInPixel(const LocalPoint& p) const;

  // Errors
  // Error in local (cm) from the masurement errors
  LocalError localError(const MeasurementPoint&, const MeasurementError&) const override;
  // Errors in pitch units from localpoint error (in cm)
  MeasurementError measurementError(const LocalPoint&, const LocalError&) const override;

  //-------------------------------------------------------------
  // Transform LocalPoint to channel. Call pixel()
  int channel(const LocalPoint& lp) const override {
    std::pair<float, float> p = pixel(lp);
    return MTDChannelIdentifier::pixelToChannel(int(p.first), int(p.second));
  }

  //----
  // Transforms between module-local coordinates and pixel-local coordinates
  // don't need a transform for errors, same units
  LocalPoint moduleToPixelLocalPoint(const LocalPoint& mlp) const {
    std::pair<float, float> p = pixel(mlp);
    return LocalPoint(mlp.x() - (m_xoffset + (int(p.first) + 0.5f) * m_pitchx),
                      mlp.y() - (m_yoffset + (int(p.second) + 0.5f) * m_pitchy),
                      mlp.z());
  }
  LocalPoint pixelToModuleLocalPoint(const LocalPoint& plp, int row, int col) const {
    return LocalPoint(
        plp.x() + (m_xoffset + (row + 0.5f) * m_pitchx), plp.y() + (m_yoffset + (col + 0.5f) * m_pitchy), plp.z());
  }
  LocalPoint pixelToModuleLocalPoint(const LocalPoint& plp, int channel) const {
    std::pair<int, int> p = MTDChannelIdentifier::channelToPixel(channel);
    return pixelToModuleLocalPoint(plp, p.first, p.second);
  }

  //-------------------------------------------------------------
  // Return the BIG pixel information for a given pixel
  bool isItBigPixelInX(const int ixbin) const override { return false; }

  bool isItBigPixelInY(const int iybin) const override { return false; }

  //-------------------------------------------------------------
  // Return BIG pixel flag in a given pixel range
  bool containsBigPixelInX(int ixmin, int ixmax) const override { return false; }

  bool containsBigPixelInY(int iymin, int iymax) const override { return false; }

  // Check whether the pixel is at the edge of the module
  bool isItEdgePixelInX(int ixbin) const override { return ((ixbin == 0) | (ixbin == (m_nrows - 1))); }

  bool isItEdgePixelInY(int iybin) const override { return ((iybin == 0) | (iybin == (m_ncols - 1))); }

  bool isItEdgePixel(int ixbin, int iybin) const override {
    return (isItEdgePixelInX(ixbin) || isItEdgePixelInY(iybin));
  }

  //-------------------------------------------------------------
  // Transform measurement to local coordinates individually in each dimension
  //
  float localX(const float mpX) const override;
  float localY(const float mpY) const override;

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
  float xoffset() const { return m_xoffset; }
  float yoffset() const { return m_yoffset; }
  float gapxInterpad() const { return m_GAPxInterpad; }  // Value returned in cm
  float gapyInterpad() const { return m_GAPyInterpad; }  // Value returned in cm
  float gapxBorder() const { return m_GAPxBorder; }      // Value returned in cm
  float gapyBorder() const { return m_GAPyBorder; }      // Value returned in cm
  float gapxInterpadFrac() const { return m_GAPxInterpadFrac; }
  float gapyInterpadFrac() const { return m_GAPyInterpadFrac; }
  float gapxBorderFrac() const { return m_GAPxBorderFrac; }
  float gapyBorderFrac() const { return m_GAPyBorderFrac; }
  bool isBricked() const override { return false; }

private:
  float m_pitchx;
  float m_pitchy;
  float m_xoffset;
  float m_yoffset;
  int m_nrows;
  int m_ncols;
  int m_ROWS_PER_ROC;
  int m_COLS_PER_ROC;
  int m_ROCS_X;
  int m_ROCS_Y;
  float m_GAPxInterpad;
  float m_GAPxBorder;
  float m_GAPyInterpad;
  float m_GAPyBorder;
  float m_GAPxInterpadFrac;
  float m_GAPxBorderFrac;
  float m_GAPyInterpadFrac;
  float m_GAPyBorderFrac;
};

#endif
