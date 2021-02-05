#ifndef CondFormats_PPSObjects_PPSPixelTopology_h
#define CondFormats_PPSObjects_PPSPixelTopology_h
// -*- C++ -*-
//
// Package:    PPSObjects
// Class:      PPSPixelTopology
//
/**\class PPSPixelTopology PPSPixelTopology.h CondFormats/PPSObjects/src/PPSPixelTopology.cc

 Description: Internal topology of PPS detectors

 Implementation:
     <Notes on implementation>
*/
//

#include "CondFormats/Serialization/interface/Serializable.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/PPSObjects/interface/CTPPSPixelIndices.h"
#include <cmath>

class PPSPixelTopology {
public:
  // Constructor
  PPSPixelTopology();
  // Destructor
  ~PPSPixelTopology();

  class PixelInfo {
  public:
    PixelInfo(double lower_simX_border,
              double higher_simX_border,
              double lower_simY_border,
              double higher_simY_border,
              double eff_factor,
              unsigned short pixel_row_no,
              unsigned short pixel_col_no)
        : lower_simX_border_(lower_simX_border),
          higher_simX_border_(higher_simX_border),
          lower_simY_border_(lower_simY_border),
          higher_simY_border_(higher_simY_border),
          eff_factor_(eff_factor),
          pixel_row_no_(pixel_row_no),
          pixel_col_no_(pixel_col_no)
    //,
    //      pixel_index_(pixel_col_no * PPSPixelTopology::no_of_pixels_simX_ + pixel_row_no)
    {}

    inline double higherSimXBorder() const { return higher_simX_border_; }
    inline double lowerSimXBorder() const { return lower_simX_border_; }
    inline double higherSimYBorder() const { return higher_simY_border_; }
    inline double lowerSimYBorder() const { return lower_simY_border_; }
    inline double effFactor() const { return eff_factor_; }
    inline unsigned short pixelRowNo() const { return pixel_row_no_; }
    inline unsigned short pixelColNo() const { return pixel_col_no_; }
    //    inline unsigned short pixelIndex() const { return pixel_index_; }

  private:
    double lower_simX_border_;
    double higher_simX_border_;
    double lower_simY_border_;
    double higher_simY_border_;
    double eff_factor_;
    unsigned short pixel_row_no_;
    unsigned short pixel_col_no_;
    //    unsigned short pixel_index_;
    COND_SERIALIZABLE;
  };

  unsigned short pixelIndex(PixelInfo pI) const;
  bool isPixelHit(float xLocalCoordinate, float yLocalCoordinate, bool is3x2) const;
  PixelInfo getPixelsInvolved(double x, double y, double sigma, double& hit_pos_x, double& hit_pos_y) const;

  void pixelRange(
      unsigned int arow, unsigned int acol, double& lower_x, double& higher_x, double& lower_y, double& higher_y) const;

  // Getters

  std::string getRunType() const;
  double getPitchSimY() const;
  double getPitchSimX() const;
  double getThickness() const;
  unsigned short getNoPixelsSimX() const;
  unsigned short getNoPixelsSimY() const;
  unsigned short getNoPixels() const;
  double getSimXWidth() const;
  double getSimYWidth() const;
  double getDeadEdgeWidth() const;
  double getActiveEdgeSigma() const;
  double getPhysActiveEdgeDist() const;
  double getActiveEdgeX() const;
  double getActiveEdgeY() const;

  // Setters

  void setRunType(std::string rt);
  void setPitchSimY(double psy);
  void setPitchSimX(double psx);
  void setThickness(double tss);
  void setNoPixelsSimX(unsigned short npx);
  void setNoPixelsSimY(unsigned short npy);
  void setNoPixels(unsigned short np);
  void setSimXWidth(double sxw);
  void setSimYWidth(double syw);
  void setDeadEdgeWidth(double dew);
  void setActiveEdgeSigma(double aes);
  void setPhysActiveEdgeDist(double pae);
  void setActiveEdgeX(double aex);
  void setActiveEdgeY(double aey);

  void printInfo(std::stringstream& s);

private:
  /*
Geometrical and topological information on RPix silicon detector.
Uses coordinate a frame with origin in the center of the wafer.
*/

  double activeEdgeFactor(double x, double y) const;
  double distanceFromTopActiveEdge(double x, double y) const;
  double distanceFromBottomActiveEdge(double x, double y) const;
  double distanceFromRightActiveEdge(double x, double y) const;
  double distanceFromLeftActiveEdge(double x, double y) const;
  unsigned int row(double x) const;
  unsigned int col(double y) const;
  void rowCol2Index(unsigned int arow, unsigned int acol, unsigned int& index) const;
  void index2RowCol(unsigned int& arow, unsigned int& acol, unsigned int index) const;

  std::string runType_;
  double pitch_simY_;
  double pitch_simX_;
  double thickness_;
  unsigned short no_of_pixels_simX_;
  unsigned short no_of_pixels_simY_;
  unsigned short no_of_pixels_;
  double simX_width_;
  double simY_width_;
  double dead_edge_width_;
  double active_edge_sigma_;
  double phys_active_edge_dist_;

  double active_edge_x_;
  double active_edge_y_;

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream&, PPSPixelTopology);

#endif
