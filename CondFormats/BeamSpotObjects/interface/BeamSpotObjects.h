#ifndef BEAMSPOTOBJECTS_H
#define BEAMSPOTOBJECTS_H
/** \class BeamSpotObjects
 *
 * Reconstructed beam spot object. It provides position, error, and
 * width of the beam position.
 *
 * \author Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
 *
 * \version $Id: BeamSpotObjects.h,v 1.9 2009/03/26 18:39:42 yumiceva Exp $
 *
 */

#include "CondFormats/Serialization/interface/Serializable.h"

#include <cmath>
#include <sstream>
#include <cstring>

class BeamSpotObjects {
public:
  /// default constructor
  BeamSpotObjects() : sigmaZ_(0), beamwidthX_(0), beamwidthY_(0), dxdz_(0), dydz_(0), type_(-1) {
    beamwidthXError_ = 0;
    beamwidthYError_ = 0;
    emittanceX_ = 0;
    emittanceY_ = 0;
    betaStar_ = 0;
    std::memset(position_, 0, sizeof position_);
    std::memset(covariance_, 0, sizeof covariance_);
  }

  virtual ~BeamSpotObjects() {}

  /// set XYZ position
  void setPosition(double x, double y, double z) {
    position_[0] = x;
    position_[1] = y;
    position_[2] = z;
  };
  /// set sigma Z, RMS bunch length
  void setSigmaZ(double val) { sigmaZ_ = val; }
  /// set dxdz slope, crossing angle
  void setdxdz(double val) { dxdz_ = val; }
  /// set dydz slope, crossing angle in XZ
  void setdydz(double val) { dydz_ = val; }
  /// set average transverse beam width X
  void setBeamWidthX(double val) { beamwidthX_ = val; }
  /// set average transverse beam width Y
  void setBeamWidthY(double val) { beamwidthY_ = val; }
  /// set beam width X error
  void setBeamWidthXError(double val) { beamwidthXError_ = val; }
  /// set beam width Y error
  void setBeamWidthYError(double val) { beamwidthYError_ = val; }
  /// set i,j element of the full covariance matrix 7x7
  void setCovariance(int i, int j, double val) { covariance_[i][j] = val; }
  /// set beam type
  void setType(int type) { type_ = type; }
  /// set emittance
  void setEmittanceX(double val) { emittanceX_ = val; }
  /// set emittance
  void setEmittanceY(double val) { emittanceY_ = val; }
  /// set beta star
  void setBetaStar(double val) { betaStar_ = val; }

  /// get X beam position
  double x() const { return position_[0]; }
  /// get Y beam position
  double y() const { return position_[1]; }
  /// get Z beam position
  double z() const { return position_[2]; }
  /// get sigma Z, RMS bunch length
  double sigmaZ() const { return sigmaZ_; }
  /// get average transverse beam width
  double beamWidthX() const { return beamwidthX_; }
  /// get average transverse beam width
  double beamWidthY() const { return beamwidthY_; }
  /// get dxdz slope, crossing angle in XZ
  double dxdz() const { return dxdz_; }
  /// get dydz slope, crossing angle in YZ
  double dydz() const { return dydz_; }
  /// get i,j element of the full covariance matrix 7x7
  double covariance(int i, int j) const { return covariance_[i][j]; }
  /// get X beam position Error
  double xError() const { return sqrt(covariance_[0][0]); }
  /// get Y beam position Error
  double yError() const { return sqrt(covariance_[1][1]); }
  /// get Z beam position Error
  double zError() const { return sqrt(covariance_[2][2]); }
  /// get sigma Z, RMS bunch length Error
  double sigmaZError() const { return sqrt(covariance_[3][3]); }
  /// get average transverse beam width error ASSUME the same for X and Y
  double beamWidthXError() const { return sqrt(covariance_[6][6]); }
  /// get average transverse beam width error X = Y
  double beamWidthYError() const { return sqrt(covariance_[6][6]); }
  /// get dxdz slope, crossing angle in XZ Error
  double dxdzError() const { return sqrt(covariance_[4][4]); }
  /// get dydz slope, crossing angle in YZ Error
  double dydzError() const { return sqrt(covariance_[5][5]); }
  /// get beam type
  int beamType() const { return type_; }
  /// get emittance
  double emittanceX() const { return emittanceX_; }
  /// get emittance
  double emittanceY() const { return emittanceY_; }
  /// get beta star
  double betaStar() const { return betaStar_; }

  /// print beam spot parameters
  void print(std::stringstream& ss) const;

protected:
  double position_[3];
  double sigmaZ_;
  double beamwidthX_;
  double beamwidthY_;
  double beamwidthXError_;
  double beamwidthYError_;
  double dxdz_;
  double dydz_;
  double covariance_[7][7];
  int type_;
  double emittanceX_;
  double emittanceY_;
  double betaStar_;

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream&, BeamSpotObjects beam);

#endif
