#ifndef Alignment_MuonAlignmentAlgorithms_MuonResiduals5DOFFitter_H
#define Alignment_MuonAlignmentAlgorithms_MuonResiduals5DOFFitter_H

/** \class MuonResiduals5DOFFitter
 *  $Date: Fri Apr 17 15:29:54 CDT 2009
 *  $Revision: 1.3 $ 
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h"

class MuonResiduals5DOFFitter: public MuonResidualsFitter {
public:
  enum {
    kAlignX = 0,
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

  MuonResiduals5DOFFitter(int residualsModel, int minHits, bool weightAlignment=true): MuonResidualsFitter(residualsModel, minHits, weightAlignment) {};

  int type() const { return MuonResidualsFitter::k5DOF; };

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
};

double MuonResiduals5DOFFitter_resid(double delta_x, double delta_z, double delta_phix, double delta_phiy, double delta_phiz, double track_x, double track_y, double track_dxdz, double track_dydz, double alpha, double resslope);
double MuonResiduals5DOFFitter_resslope(double delta_x, double delta_z, double delta_phix, double delta_phiy, double delta_phiz, double track_x, double track_y, double track_dxdz, double track_dydz);

#endif // Alignment_MuonAlignmentAlgorithms_MuonResiduals5DOFFitter_H
