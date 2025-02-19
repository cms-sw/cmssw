#ifndef Alignment_MuonAlignmentAlgorithms_MuonResidualsPositionFitter_H
#define Alignment_MuonAlignmentAlgorithms_MuonResidualsPositionFitter_H

/** \class MuonResidualsPositionFitter
 *  $Date: 2011/10/12 23:18:05 $
 *  $Revision: 1.9 $
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

  MuonResidualsPositionFitter(int residualsModel, int minHits, int useResiduals, bool weightAlignment=true): MuonResidualsFitter(residualsModel, minHits, useResiduals, weightAlignment) {}

  int type() const { return MuonResidualsFitter::kPositionFitter; }

  int npar() {
    if (residualsModel() == kPureGaussian || residualsModel() == kGaussPowerTails) return kNPar - 1;
    else if (residualsModel() == kPowerLawTails) return kNPar;
    else if (residualsModel() == kROOTVoigt) return kNPar;
    else assert(false);
  }
  int ndata() { return kNData; }

  bool fit(Alignable *ali);
  double sumofweights() { return numResiduals(); }
  double plot(std::string name, TFileDirectory *dir, Alignable *ali);

protected:
  void inform(TMinuit *tMinuit);
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonResidualsPositionFitter_H
