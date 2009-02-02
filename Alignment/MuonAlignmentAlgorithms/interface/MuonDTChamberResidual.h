#ifndef Alignment_MuonAlignmentAlgorithms_MuonDTChamberResidual_H
#define Alignment_MuonAlignmentAlgorithms_MuonDTChamberResidual_H

/** \class MuonDTChamberResidual
 *  $Date: Sat Jan 24 18:33:33 CST 2009 $
 *  $Revision: 1.0 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonChamberResidual.h"

class MuonDTChamberResidual: public MuonChamberResidual {
public:
  MuonDTChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator, DetId chamberId, AlignableDetOrUnitPtr chamberAlignable)
    : MuonChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable)

    , m_SL13_N(0)
    , m_SL13_denom(0.)
    , m_SL13_residglobal(0.)
    , m_SL13_residx(0.)
    , m_SL13_zpos(0.)
    , m_SL13_Rpos(0.)
    , m_SL13_phipos(0.)
    , m_SL13_phimin(0.)
    , m_SL13_phimax(0.)
    , m_SL13_localxpos(0.)
    , m_SL13_localypos(0.)

    , m_SL2_N(0)
    , m_SL2_denom(0.)
    , m_SL2_residglobal(0.)
    , m_SL2_residy(0.)
    , m_SL2_zpos(0.)
    , m_SL2_Rpos(0.)
    , m_SL2_phipos(0.)
    , m_SL2_phimin(0.)
    , m_SL2_phimax(0.)
    , m_SL2_localxpos(0.)
    , m_SL2_localypos(0.)

    , m_residz(0.)
    , m_residz_denom(0.)
    , m_residphix(0.)
    , m_residphix_denom(0.)
    , m_residphiy(0.)
    , m_residphiy_denom(0.)
    , m_residphiz(0.)
    , m_residphiz_denom(0.)
  {};

  ~MuonDTChamberResidual() {};

  enum {
    kSuperLayer13,
    kSuperLayer2,
    kAllSuperLayers
  };
  
  void addResidual(const TrajectoryStateOnSurface *tsos, const TransientTrackingRecHit *hit);

  bool isRphiValid() { return (m_SL13_N > 1  &&  fabs(m_SL13_phimax - m_SL13_phimin) < 1.); };
  bool isZValid() { return (m_SL2_N > 1  &&  fabs(m_SL2_phimax - m_SL2_phimin) < 1.); };
  bool isRValid() { return false; };
  int rphiHits() { return m_SL13_N; };
  int zHits() { return m_SL2_N; };
  int rHits() { return 0; };

  double globalRphiResidual() { return m_SL13_residglobal / m_SL13_denom; };
  double globalZResidual()    { return m_SL2_residglobal / m_SL2_denom; };
  double globalRResidual()    { return 0.; };
  double x_residual()         { return m_SL13_residx / m_SL13_denom; };
  double y_residual()         { return m_SL2_residy / m_SL2_denom; };
  double z_residual()         { return m_residz / m_residz_denom; };
  double phix_residual()      { return m_residphix / m_residphix_denom; };
  double phiy_residual()      { return m_residphiy / m_residphiy_denom; };
  double phiz_residual()      { return m_residphiz / m_residphiz_denom; };

  double phi_position(int which) {
    if (which == kSuperLayer13) return m_SL13_phipos / m_SL13_denom;
    else if (which == kSuperLayer2) return m_SL2_phipos / m_SL2_denom;
    else if (which == kAllSuperLayers) return (m_SL13_phipos + m_SL2_phipos) / (m_SL13_denom + m_SL2_denom);
    else assert(false);
  };

  double z_position(int which) {
    if (which == kSuperLayer13) return m_SL13_zpos / m_SL13_denom;
    else if (which == kSuperLayer2) return m_SL2_zpos / m_SL2_denom;
    else if (which == kAllSuperLayers) return (m_SL13_zpos + m_SL2_zpos) / (m_SL13_denom + m_SL2_denom);
    else assert(false);
  };

  double R_position(int which) {
    if (which == kSuperLayer13) return m_SL13_Rpos / m_SL13_denom;
    else if (which == kSuperLayer2) return m_SL2_Rpos / m_SL2_denom;
    else if (which == kAllSuperLayers) return (m_SL13_Rpos + m_SL2_Rpos) / (m_SL13_denom + m_SL2_denom);
    else assert(false);
  };

  double localx_position(int which) {
    if (which == kSuperLayer13) return m_SL13_localxpos / m_SL13_denom;
    else if (which == kSuperLayer2) return m_SL2_localxpos / m_SL2_denom;
    else if (which == kAllSuperLayers) return (m_SL13_localxpos + m_SL2_localxpos) / (m_SL13_denom + m_SL2_denom);
    else assert(false);
  };

  double localy_position(int which) {
    if (which == kSuperLayer13) return m_SL13_localypos / m_SL13_denom;
    else if (which == kSuperLayer2) return m_SL2_localypos / m_SL2_denom;
    else if (which == kAllSuperLayers) return (m_SL13_localypos + m_SL2_localypos) / (m_SL13_denom + m_SL2_denom);
    else assert(false);
  };

private:
  int m_SL13_N;
  double m_SL13_denom;
  double m_SL13_residglobal;
  double m_SL13_residx;
  double m_SL13_residphiy;
  double m_SL13_zpos;
  double m_SL13_Rpos;
  double m_SL13_phipos;
  double m_SL13_phimin;
  double m_SL13_phimax;
  double m_SL13_localxpos;
  double m_SL13_localypos;

  int m_SL2_N;
  double m_SL2_denom;
  double m_SL2_residglobal;
  double m_SL2_residy;
  double m_SL2_residphix;
  double m_SL2_zpos;
  double m_SL2_Rpos;
  double m_SL2_phipos;
  double m_SL2_phimin;
  double m_SL2_phimax;
  double m_SL2_localxpos;
  double m_SL2_localypos;

  double m_residz;
  double m_residz_denom;
  double m_residphix;
  double m_residphix_denom;
  double m_residphiy;
  double m_residphiy_denom;
  double m_residphiz;
  double m_residphiz_denom;
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonDTChamberResidual_H
