#ifndef Alignment_MuonAlignmentAlgorithms_MuonResiduals1DOFFitter_H
#define Alignment_MuonAlignmentAlgorithms_MuonResiduals1DOFFitter_H

/** \class MuonResiduals1DOFFitter
 *  $Date: Fri Apr 17 16:09:40 CDT 2009
 *  $Revision: 1.4 $ 
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h"

class MuonResiduals1DOFFitter : public MuonResidualsFitter {
public:
  enum { kAlign = 0, kSigma, kGamma, kNPar };

  enum { kResid = 0, kRedChi2, kNData };

  MuonResiduals1DOFFitter(int residualsModel, int minHits, int useResiduals, bool weightAlignment = true)
      : MuonResidualsFitter(residualsModel, minHits, useResiduals, weightAlignment) {}

  int type() const override { return MuonResidualsFitter::k1DOF; }

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

  double sumofweights() override;
  bool fit(Alignable *ali) override;
  double plot(std::string name, TFileDirectory *dir, Alignable *ali) override;

protected:
  void inform(TMinuit *tMinuit) override;
};

#endif  // Alignment_MuonAlignmentAlgorithms_MuonResiduals1DOFFitter_H
