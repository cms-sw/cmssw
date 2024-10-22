/****************************************************************************
* Authors:
*	Jan Kašpar (jan.kaspar@gmail.com)
****************************************************************************/

#ifndef CalibPPS_AlignmentRelative_AlignmentResult_h
#define CalibPPS_AlignmentRelative_AlignmentResult_h

/// \brief Result of CTPPS track-based alignment
class AlignmentResult {
protected:
  /// shift in read-out directions and along beam, mm
  /// "_unc" denotes the shift uncertainties
  double sh_r1, sh_r1_unc;
  double sh_r2, sh_r2_unc;
  double sh_z, sh_z_unc;

  /// rotation about beam axis, rad
  double rot_z;
  double rot_z_unc;

public:
  AlignmentResult(double _sh_r1 = 0.,
                  double _sh_r1_e = 0.,
                  double _sh_r2 = 0.,
                  double _sh_r2_e = 0.,
                  double _sh_z = 0.,
                  double _sh_z_e = 0.,
                  double _rot_z = 0.,
                  double _rot_z_e = 0.)
      : sh_r1(_sh_r1),
        sh_r1_unc(_sh_r1_e),
        sh_r2(_sh_r2),
        sh_r2_unc(_sh_r2_e),
        sh_z(_sh_z),
        sh_z_unc(_sh_z_e),
        rot_z(_rot_z),
        rot_z_unc(_rot_z_e) {}

  inline double getShR1() const { return sh_r1; }
  inline void setShR1(const double &v) { sh_r1 = v; }

  inline double getShR1Unc() const { return sh_r1_unc; }
  inline void setShR1Unc(const double &v) { sh_r1_unc = v; }

  inline double getShR2() const { return sh_r2; }
  inline void setShR2(const double &v) { sh_r2 = v; }

  inline double getShR2Unc() const { return sh_r2_unc; }
  inline void setShR2Unc(const double &v) { sh_r2_unc = v; }

  inline double getShZ() const { return sh_z; }
  inline void setShZ(const double &v) { sh_z = v; }

  inline double getShZUnc() const { return sh_z_unc; }
  inline void setShZUnc(const double &v) { sh_z_unc = v; }

  inline double getRotZ() const { return rot_z; }
  inline void setRotZ(const double &v) { rot_z = v; }

  inline double getRotZUnc() const { return rot_z_unc; }
  inline void setRotZUnc(const double &v) { rot_z_unc = v; }
};

#endif
