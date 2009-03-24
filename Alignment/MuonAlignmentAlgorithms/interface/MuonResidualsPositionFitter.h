#ifndef Alignment_MuonAlignmentAlgorithms_MuonResidualsPositionFitter_H
#define Alignment_MuonAlignmentAlgorithms_MuonResidualsPositionFitter_H

/** \class MuonResidualsPositionFitter
 *  $Date: 2009/03/23 11:50:53 $
 *  $Revision: 1.3 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h"

class MuonResidualsPositionFitter: public MuonResidualsFitter {
public:
  enum {
    kPosition = 0,
    kZpos,
    kPhiz,
    kScattering,
    kSigma,
    kGamma,
    kNPar
  };

  enum {
    kResidual = 0,
    kAngleError,
    kTrackAngle,
    kTrackPosition,
    kNData
  };

  MuonResidualsPositionFitter(int residualsModel, int minHitsPerRegion): MuonResidualsFitter(residualsModel, minHitsPerRegion) {};

  int npar() {
    if (residualsModel() == kPureGaussian) return kNPar - 1;
    else if (residualsModel() == kPowerLawTails) return kNPar;
    else assert(false);
  };
  int ndata() { return kNData; };

  bool fit(double phiValue);
  void plot(double phiValue, std::string name, TFileDirectory *dir);
  double redchi2(double phiValue, std::string name, TFileDirectory *dir, bool write=false, int bins=100, double low=-5., double high=5.);

protected:
  void inform(TMinuit *tMinuit);
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonResidualsPositionFitter_H
