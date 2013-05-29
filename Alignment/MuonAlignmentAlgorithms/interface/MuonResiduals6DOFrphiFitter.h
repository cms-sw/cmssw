#ifndef Alignment_MuonAlignmentAlgorithms_MuonResiduals6DOFrphiFitter_H
#define Alignment_MuonAlignmentAlgorithms_MuonResiduals6DOFrphiFitter_H

/** \class MuonResiduals6DOFrphiFitter
 *  $Date: Thu Apr 16 21:29:15 CDT 2009
 *  $Revision: 1.3 $ 
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

class MuonResiduals6DOFrphiFitter: public MuonResidualsFitter {
public:
  enum {
    kAlignX = 0,
    kAlignY,
    kAlignZ,
    kAlignPhiX,
    kAlignPhiY,
    kAlignPhiZ,
    kResidSigma,
    kResSlopeSigma,
    kAlpha,
    kResidGamma,
    kResSlopeGamma,
    kNPar
  };

  enum {
    kResid = 0,
    kResSlope,
    kPositionX,
    kPositionY,
    kAngleX,
    kAngleY,
    kRedChi2,
    kNData
  };

  MuonResiduals6DOFrphiFitter(int residualsModel, int minHits, const CSCGeometry *cscGeometry, bool weightAlignment=true): MuonResidualsFitter(residualsModel, minHits, weightAlignment), m_cscGeometry(cscGeometry) {};

  int type() const { return MuonResidualsFitter::k6DOFrphi; };

  int npar() {
    if (residualsModel() == kPureGaussian || residualsModel() == kGaussPowerTails) return kNPar - 2;
    else if (residualsModel() == kPowerLawTails) return kNPar;
    else if (residualsModel() == kROOTVoigt) return kNPar;
    else assert(false);
  };
  int ndata() { return kNData; };

  double sumofweights();
  bool fit(Alignable *ali);
  double plot(std::string name, TFileDirectory *dir, Alignable *ali);

protected:
  void inform(TMinuit *tMinuit);

private:
  const CSCGeometry *m_cscGeometry;
};

double MuonResiduals6DOFrphiFitter_residual(double delta_x, double delta_y, double delta_z, double delta_phix, double delta_phiy, double delta_phiz, double track_x, double track_y, double track_dxdz, double track_dydz, double R, double alpha, double resslope);
double MuonResiduals6DOFrphiFitter_resslope(double delta_x, double delta_y, double delta_z, double delta_phix, double delta_phiy, double delta_phiz, double track_x, double track_y, double track_dxdz, double track_dydz, double R);

#endif // Alignment_MuonAlignmentAlgorithms_MuonResiduals6DOFrphiFitter_H
