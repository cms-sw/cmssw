#ifndef Alignment_MuonAlignmentAlgorithms_MuonResidualsAngleFitter_H
#define Alignment_MuonAlignmentAlgorithms_MuonResidualsAngleFitter_H

/** \class MuonResidualsAngleFitter
 *  $Date: 2011/10/12 23:18:05 $
 *  $Revision: 1.11 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h"

class MuonResidualsAngleFitter: public MuonResidualsFitter {
public:
  enum {
    kAngle = 0,
    kXControl,
    kYControl,
    kSigma,
    kGamma,
    kNPar
  };

  enum {
    kResidual = 0,
    kXAngle,
    kYAngle,
    kNData
  };

  MuonResidualsAngleFitter(int residualsModel, int minHitsPerRegion, int useResiduals, bool weightAlignment=true): MuonResidualsFitter(residualsModel, minHitsPerRegion, useResiduals, weightAlignment) {}

  int type() const { return MuonResidualsFitter::kAngleFitter; }

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

#endif // Alignment_MuonAlignmentAlgorithms_MuonResidualsAngleFitter_H
