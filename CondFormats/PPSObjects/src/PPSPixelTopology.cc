#include "CondFormats/PPSObjects/interface/PPSPixelTopology.h"

// Constructors

PPSPixelTopology::PPSPixelTopology()
    : runType_(""),
      pitch_simY_(0.),
      pitch_simX_(0.),
      thickness_(0.),
      no_of_pixels_simX_(0.),
      no_of_pixels_simY_(0.),
      no_of_pixels_(0.),
      simX_width_(0.),
      simY_width_(0.),
      dead_edge_width_(0.),
      active_edge_sigma_(0.),
      phys_active_edge_dist_(0.),
      active_edge_x_(0.),
      active_edge_y_(0.) {}

// Destructor
PPSPixelTopology::~PPSPixelTopology() {}

unsigned short PPSPixelTopology::pixelIndex(PixelInfo pI) const {
  return no_of_pixels_simX_ * pI.pixelColNo() + pI.pixelRowNo();
}

bool PPSPixelTopology::isPixelHit(float xLocalCoordinate, float yLocalCoordinate, bool is3x2 = true) const {
  // check hit fiducial boundaries
  const double xModuleSize = 2 * ((no_of_pixels_simX_ / 2. + 1) * pitch_simX_ + dead_edge_width_);
  if (xLocalCoordinate < -xModuleSize / 2. || xLocalCoordinate > xModuleSize / 2.)
    return false;

  const double yModuleSize = (no_of_pixels_simY_ + 4.) * pitch_simY_ + 2. * dead_edge_width_;
  const double y2x2top = no_of_pixels_simY_ / 6. * pitch_simY_ + dead_edge_width_;
  if (is3x2 && (yLocalCoordinate < -yModuleSize / 2. || yLocalCoordinate > yModuleSize / 2.))
    return false;
  if (!is3x2 && (runType_ == "Run2") && (yLocalCoordinate < -yModuleSize / 2. || yLocalCoordinate > y2x2top))
    return false;
  if (!is3x2 && (runType_ == "Run3") && (yLocalCoordinate < -yModuleSize / 2. || yLocalCoordinate > yModuleSize / 2.))
    return false;

  return true;
}

PPSPixelTopology::PixelInfo PPSPixelTopology::getPixelsInvolved(
    double x, double y, double sigma, double& hit_pos_x, double& hit_pos_y) const {
  //hit position wrt the bottom left corner of the sensor (-8.3, -12.2) in sensor view, rocs behind
  hit_pos_x = x + simX_width_ / 2.;
  hit_pos_y = y + simY_width_ / 2.;
  if (!(hit_pos_x * hit_pos_y > 0))
    throw cms::Exception("PPSPixelTopology") << "pixel out of reference frame";

  double hit_factor = activeEdgeFactor(x, y);

  unsigned int interested_row = row(x);
  unsigned int interested_col = col(y);
  double low_pixel_range_x, high_pixel_range_x, low_pixel_range_y, high_pixel_range_y;
  pixelRange(
      interested_row, interested_col, low_pixel_range_x, high_pixel_range_x, low_pixel_range_y, high_pixel_range_y);

  return PPSPixelTopology::PixelInfo(low_pixel_range_x,
                                     high_pixel_range_x,
                                     low_pixel_range_y,
                                     high_pixel_range_y,
                                     hit_factor,
                                     interested_row,
                                     interested_col);
}

void PPSPixelTopology::pixelRange(
    unsigned int arow, unsigned int acol, double& lower_x, double& higher_x, double& lower_y, double& higher_y) const {
  // x and y in the system  of Geant4 SIMULATION
  arow = (2 * rpixValues::ROCSizeInX - 1) - arow;
  if (arow > (2 * rpixValues::ROCSizeInX - 1) || acol > (3 * rpixValues::ROCSizeInY - 1))
    throw cms::Exception("PPSPixelTopology") << "pixel rows or columns exceeding limits";

  // rows (x segmentation)
  if (arow == 0) {
    lower_x = dead_edge_width_ - phys_active_edge_dist_;  // 50 um
    higher_x = dead_edge_width_ + pitch_simX_;            // 300 um
  } else if (arow <= (rpixValues::ROCSizeInX - 2)) {
    lower_x = dead_edge_width_ + arow * pitch_simX_;
    higher_x = dead_edge_width_ + (arow + 1) * pitch_simX_;
  } else if (arow == (rpixValues::ROCSizeInX - 1)) {
    lower_x = dead_edge_width_ + arow * pitch_simX_;
    higher_x = dead_edge_width_ + (arow + 2) * pitch_simX_;
  } else if (arow == rpixValues::ROCSizeInX) {
    lower_x = dead_edge_width_ + (arow + 1) * pitch_simX_;
    higher_x = dead_edge_width_ + (arow + 3) * pitch_simX_;
  } else if (arow <= (2 * rpixValues::ROCSizeInX - 2)) {
    lower_x = dead_edge_width_ + (arow + 2) * pitch_simX_;
    higher_x = dead_edge_width_ + (arow + 3) * pitch_simX_;
  } else if (arow == (2 * rpixValues::ROCSizeInX - 1)) {
    lower_x = dead_edge_width_ + (arow + 2) * pitch_simX_;
    higher_x = dead_edge_width_ + (arow + 3) * pitch_simX_ + phys_active_edge_dist_;
  }
  // columns (y segmentation)
  if (acol == 0) {
    lower_y = dead_edge_width_ - phys_active_edge_dist_;  // 50 um
    higher_y = dead_edge_width_ + pitch_simY_;            // 350 um
  } else if (acol <= (rpixValues::ROCSizeInY - 2)) {
    lower_y = dead_edge_width_ + acol * pitch_simY_;
    higher_y = dead_edge_width_ + (acol + 1) * pitch_simY_;
  } else if (acol == (rpixValues::ROCSizeInY - 1)) {
    lower_y = dead_edge_width_ + acol * pitch_simY_;
    higher_y = dead_edge_width_ + (acol + 2) * pitch_simY_;
  } else if (acol == rpixValues::ROCSizeInY) {
    lower_y = dead_edge_width_ + (acol + 1) * pitch_simY_;
    higher_y = dead_edge_width_ + (acol + 3) * pitch_simY_;
  } else if (acol <= (2 * rpixValues::ROCSizeInY - 2)) {
    lower_y = dead_edge_width_ + (acol + 2) * pitch_simY_;
    higher_y = dead_edge_width_ + (acol + 3) * pitch_simY_;
  } else if (acol == (2 * rpixValues::ROCSizeInY - 1)) {
    lower_y = dead_edge_width_ + (acol + 2) * pitch_simY_;
    higher_y = dead_edge_width_ + (acol + 4) * pitch_simY_;
  } else if (acol == (2 * rpixValues::ROCSizeInY)) {
    lower_y = dead_edge_width_ + (acol + 3) * pitch_simY_;
    higher_y = dead_edge_width_ + (acol + 5) * pitch_simY_;
  } else if (acol <= (3 * rpixValues::ROCSizeInY - 2)) {
    lower_y = dead_edge_width_ + (acol + 4) * pitch_simY_;
    higher_y = dead_edge_width_ + (acol + 5) * pitch_simY_;
  } else if (acol == (3 * rpixValues::ROCSizeInY - 1)) {
    lower_y = dead_edge_width_ + (acol + 4) * pitch_simY_;
    higher_y = dead_edge_width_ + (acol + 5) * pitch_simY_ + phys_active_edge_dist_;
  }

  lower_x = lower_x - simX_width_ / 2.;
  lower_y = lower_y - simY_width_ / 2.;
  higher_x = higher_x - simX_width_ / 2.;
  higher_y = higher_y - simY_width_ / 2.;
}

double PPSPixelTopology::activeEdgeFactor(double x, double y) const {
  const double inv_sigma = 1. / active_edge_sigma_;  // precaching
  const double topEdgeFactor = std::erf(-distanceFromTopActiveEdge(x, y) * inv_sigma) * 0.5 + 0.5;
  const double bottomEdgeFactor = std::erf(-distanceFromBottomActiveEdge(x, y) * inv_sigma) * 0.5 + 0.5;
  const double rightEdgeFactor = std::erf(-distanceFromRightActiveEdge(x, y) * inv_sigma) * 0.5 + 0.5;
  const double leftEdgeFactor = std::erf(-distanceFromLeftActiveEdge(x, y) * inv_sigma) * 0.5 + 0.5;

  const double aEF = topEdgeFactor * bottomEdgeFactor * rightEdgeFactor * leftEdgeFactor;

  if (aEF > 1.)
    throw cms::Exception("PPSPixelTopology") << " pixel active edge factor > 1";

  return aEF;
}

double PPSPixelTopology::distanceFromTopActiveEdge(double x, double y) const { return (y - active_edge_y_); }
double PPSPixelTopology::distanceFromBottomActiveEdge(double x, double y) const { return (-y - active_edge_y_); }
double PPSPixelTopology::distanceFromRightActiveEdge(double x, double y) const { return (x - active_edge_x_); }
double PPSPixelTopology::distanceFromLeftActiveEdge(double x, double y) const { return (-x - active_edge_x_); }

unsigned int PPSPixelTopology::row(double x) const {
  // x in the G4 simulation system
  x = x + simX_width_ / 2.;

  // now x in the system centered in the bottom left corner of the sensor (sensor view, rocs behind)
  if (x < 0. || x > simX_width_)
    throw cms::Exception("PPSPixelTopology") << " pixel out of reference frame";

  // rows (x segmentation)
  unsigned int arow;
  if (x <= (dead_edge_width_ + pitch_simX_))
    arow = 0;
  else if (x <= (dead_edge_width_ + (rpixValues::ROCSizeInX - 1) * pitch_simX_))
    arow = int((x - dead_edge_width_ - pitch_simX_) / pitch_simX_) + 1;
  else if (x <= (dead_edge_width_ + (rpixValues::ROCSizeInX + 1) * pitch_simX_))
    arow = (rpixValues::ROCSizeInX - 1);
  else if (x <= (dead_edge_width_ + (rpixValues::ROCSizeInX + 3) * pitch_simX_))
    arow = rpixValues::ROCSizeInX;
  else if (x <= (dead_edge_width_ + (2 * rpixValues::ROCSizeInX + 2) * pitch_simX_))
    arow = int((x - dead_edge_width_ - pitch_simX_) / pitch_simX_) - 1;
  else
    arow = (2 * rpixValues::ROCSizeInX - 1);

  arow = (2 * rpixValues::ROCSizeInX - 1) - arow;
  if (arow > (2 * rpixValues::ROCSizeInX - 1))
    throw cms::Exception("PPSPixelTopology") << " pixel row number exceeding limit";

  return arow;
}

unsigned int PPSPixelTopology::col(double y) const {
  // y in the G4 simulation system
  unsigned int column;

  // columns (y segmentation)
  // now y in the system centered in the bottom left corner of the sensor (sensor view, rocs behind)
  y = y + simY_width_ / 2.;
  if (y < 0. || y > simY_width_)
    throw cms::Exception("PPSPixelTopology") << "pixel out of reference frame";

  if (y <= (dead_edge_width_ + pitch_simY_))
    column = 0;
  else if (y <= (dead_edge_width_ + (rpixValues::ROCSizeInY - 1) * pitch_simY_))
    column = int((y - dead_edge_width_ - pitch_simY_) / pitch_simY_) + 1;
  else if (y <= (dead_edge_width_ + (rpixValues::ROCSizeInY + 1) * pitch_simY_))
    column = rpixValues::ROCSizeInY - 1;
  else if (y <= (dead_edge_width_ + (rpixValues::ROCSizeInY + 3) * pitch_simY_))
    column = rpixValues::ROCSizeInY;
  else if (y <= (dead_edge_width_ + (2 * rpixValues::ROCSizeInY + 1) * pitch_simY_))
    column = int((y - dead_edge_width_ - pitch_simY_) / pitch_simY_) - 1;
  else if (y <= (dead_edge_width_ + (2 * rpixValues::ROCSizeInY + 3) * pitch_simY_))
    column = 2 * rpixValues::ROCSizeInY - 1;
  else if (y <= (dead_edge_width_ + (2 * rpixValues::ROCSizeInY + 5) * pitch_simY_))
    column = 2 * rpixValues::ROCSizeInY;
  else if (y <= (dead_edge_width_ + (3 * rpixValues::ROCSizeInY + 3) * pitch_simY_))
    column = int((y - dead_edge_width_ - pitch_simY_) / pitch_simY_) - 3;
  else
    column = (3 * rpixValues::ROCSizeInY - 1);

  return column;
}

void PPSPixelTopology::rowCol2Index(unsigned int arow, unsigned int acol, unsigned int& index) const {
  index = acol * no_of_pixels_simX_ + arow;
}

void PPSPixelTopology::index2RowCol(unsigned int& arow, unsigned int& acol, unsigned int index) const {
  acol = index / no_of_pixels_simX_;
  arow = index % no_of_pixels_simX_;
}

// Getters

std::string PPSPixelTopology::getRunType() const { return runType_; }
double PPSPixelTopology::getPitchSimY() const { return pitch_simY_; }
double PPSPixelTopology::getPitchSimX() const { return pitch_simX_; }
double PPSPixelTopology::getThickness() const { return thickness_; }
unsigned short PPSPixelTopology::getNoPixelsSimX() const { return no_of_pixels_simX_; }
unsigned short PPSPixelTopology::getNoPixelsSimY() const { return no_of_pixels_simY_; }
unsigned short PPSPixelTopology::getNoPixels() const { return no_of_pixels_; }
double PPSPixelTopology::getSimXWidth() const { return simX_width_; }
double PPSPixelTopology::getSimYWidth() const { return simY_width_; }
double PPSPixelTopology::getDeadEdgeWidth() const { return dead_edge_width_; }
double PPSPixelTopology::getActiveEdgeSigma() const { return active_edge_sigma_; }
double PPSPixelTopology::getPhysActiveEdgeDist() const { return phys_active_edge_dist_; }
double PPSPixelTopology::getActiveEdgeX() const { return active_edge_x_; }
double PPSPixelTopology::getActiveEdgeY() const { return active_edge_y_; }

// Setters

void PPSPixelTopology::setRunType(std::string rt) { runType_ = rt; }
void PPSPixelTopology::setPitchSimY(double psy) { pitch_simY_ = psy; }
void PPSPixelTopology::setPitchSimX(double psx) { pitch_simX_ = psx; }
void PPSPixelTopology::setThickness(double tss) { thickness_ = tss; }
void PPSPixelTopology::setNoPixelsSimX(unsigned short npx) { no_of_pixels_simX_ = npx; }
void PPSPixelTopology::setNoPixelsSimY(unsigned short npy) { no_of_pixels_simY_ = npy; }
void PPSPixelTopology::setNoPixels(unsigned short np) { no_of_pixels_ = np; }
void PPSPixelTopology::setSimXWidth(double sxw) { simX_width_ = sxw; }
void PPSPixelTopology::setSimYWidth(double syw) { simY_width_ = syw; }
void PPSPixelTopology::setDeadEdgeWidth(double dew) { dead_edge_width_ = dew; }
void PPSPixelTopology::setActiveEdgeSigma(double aes) { active_edge_sigma_ = aes; }
void PPSPixelTopology::setPhysActiveEdgeDist(double pae) { phys_active_edge_dist_ = pae; }
void PPSPixelTopology::setActiveEdgeX(double aex) { active_edge_x_ = aex; }
void PPSPixelTopology::setActiveEdgeY(double aey) { active_edge_y_ = aey; }

void PPSPixelTopology::printInfo(std::stringstream& s) {
  s << "\n PPS Topology parameters : \n"
    << "\n  runType_  = " << runType_ << "\n  pitch_simY_  = " << pitch_simY_ << "\n   pitch_simX_ = " << pitch_simX_
    << "\n   thickness_ = " << thickness_ << "\n   no_of_pixels_simX_ " << no_of_pixels_simX_
    << "\n   no_of_pixels_simY_ " << no_of_pixels_simY_ << "\n   no_of_pixels_ " << no_of_pixels_ << "\n   simX_width_ "
    << simX_width_ << "\n   simY_width_ " << simY_width_ << "\n   dead_edge_width_ " << dead_edge_width_
    << "\n   active_edge_sigma_ " << active_edge_sigma_ << "\n   phys_active_edge_dist_ " << phys_active_edge_dist_

    << "\n   active_edge_x_ " << active_edge_x_ << "\n   active_edge_y_ " << active_edge_y_

    << std::endl;
}

std::ostream& operator<<(std::ostream& os, PPSPixelTopology info) {
  std::stringstream ss;
  info.printInfo(ss);
  os << ss.str();
  return os;
}
