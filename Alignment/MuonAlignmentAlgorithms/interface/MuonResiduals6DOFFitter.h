#ifndef Alignment_MuonAlignmentAlgorithms_MuonResiduals6DOFFitter_H
#define Alignment_MuonAlignmentAlgorithms_MuonResiduals6DOFFitter_H

/** \class MuonResiduals6DOFFitter
 *  $Date: Thu Apr 16 14:20:58 CDT 2009
 *  $Revision: 1.3 $ 
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h"

class MuonResiduals6DOFFitter: public MuonResidualsFitter {
public:
  enum {
    kAlignX = 0,
    kAlignY,
    kAlignZ,
    kAlignPhiX,
    kAlignPhiY,
    kAlignPhiZ,
    kResidXSigma,
    kResidYSigma,
    kResSlopeXSigma,
    kResSlopeYSigma,
    kAlphaX,
    kAlphaY,
    kResidXGamma,
    kResidYGamma,
    kResSlopeXGamma,
    kResSlopeYGamma,
    kNPar
  };

  enum {
    kResidX = 0,
    kResidY,
    kResSlopeX,
    kResSlopeY,
    kPositionX,
    kPositionY,
    kAngleX,
    kAngleY,
    kRedChi2,
    kNData
  };

  MuonResiduals6DOFFitter(int residualsModel, int minHits, bool weightAlignment=true): MuonResidualsFitter(residualsModel, minHits, weightAlignment) {};

  int type() const { return MuonResidualsFitter::k6DOF; };

  int npar() {
    if (residualsModel() == kPureGaussian || residualsModel() == kGaussPowerTails) return kNPar - 4;
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

double MuonResiduals6DOFFitter_x(double delta_x, double delta_y, double delta_z, double delta_phix, double delta_phiy, double delta_phiz, double track_x, double track_y, double track_dxdz, double track_dydz, double alphax, double residual_dxdz);
double MuonResiduals6DOFFitter_y(double delta_x, double delta_y, double delta_z, double delta_phix, double delta_phiy, double delta_phiz, double track_x, double track_y, double track_dxdz, double track_dydz, double alphay, double residual_dydz);
double MuonResiduals6DOFFitter_dxdz(double delta_x, double delta_y, double delta_z, double delta_phix, double delta_phiy, double delta_phiz, double track_x, double track_y, double track_dxdz, double track_dydz);
double MuonResiduals6DOFFitter_dydz(double delta_x, double delta_y, double delta_z, double delta_phix, double delta_phiy, double delta_phiz, double track_x, double track_y, double track_dxdz, double track_dydz);

#endif // Alignment_MuonAlignmentAlgorithms_MuonResiduals6DOFFitter_H
