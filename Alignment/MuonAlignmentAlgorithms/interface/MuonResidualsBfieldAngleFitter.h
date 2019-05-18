#ifndef Alignment_MuonAlignmentAlgorithms_MuonResidualsBfieldAngleFitter_H
#define Alignment_MuonAlignmentAlgorithms_MuonResidualsBfieldAngleFitter_H

/** \class MuonResidualsBfieldAngleFitter
 *  $Date: 2010/03/12 22:23:26 $
 *  $Revision: 1.5 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h"

class MuonResidualsBfieldAngleFitter : public MuonResidualsFitter {
public:
  enum { kAngle = 0, kBfrompt, kBfrompz, kdEdx, kSigma, kGamma, kNPar };

  enum { kResidual = 0, kQoverPt, kQoverPz, kNData };

  MuonResidualsBfieldAngleFitter(int residualsModel, int minHitsPerRegion, int useResiduals, bool weightAlignment = true)
      : MuonResidualsFitter(residualsModel, minHitsPerRegion, useResiduals, weightAlignment) {}

  int type() const override { return MuonResidualsFitter::kAngleBfieldFitter; }

  int npar() override {
    if (residualsModel() == kPureGaussian || residualsModel() == kGaussPowerTails)
      return kNPar - 1;
    else if (residualsModel() == kPowerLawTails)
      return kNPar;
    else if (residualsModel() == kROOTVoigt)
      return kNPar;
    else
      assert(false);
  }
  int ndata() override { return kNData; }

  bool fit(Alignable *ali) override;
  double sumofweights() override { return numResiduals(); }
  double plot(std::string name, TFileDirectory *dir, Alignable *ali) override;

protected:
  void inform(TMinuit *tMinuit) override;
};

#endif  // Alignment_MuonAlignmentAlgorithms_MuonResidualsBfieldAngleFitter_H
