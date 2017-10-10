#ifndef Alignment_MuonAlignmentAlgorithms_MuonResiduals6DOFFitter_H
#define Alignment_MuonAlignmentAlgorithms_MuonResiduals6DOFFitter_H

/** \class MuonResiduals6DOFFitter
 *  $Date: Thu Apr 16 14:20:58 CDT 2009
 *  $Revision: 1.5 $ 
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#ifdef STANDALONE_FITTER
#include "MuonResidualsFitter.h"
#else
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h"
#endif

class TTree;

class MuonResiduals6DOFFitter: public MuonResidualsFitter
{
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
    kPz,
    kPt,
    kCharge,
    kStation,
    kWheel,
    kSector,
    kChambW,
    kChambl,
    kNData
  };

  MuonResiduals6DOFFitter(int residualsModel, int minHits, int useResiduals, bool weightAlignment=true): MuonResidualsFitter(residualsModel, minHits, useResiduals, weightAlignment) {}
  ~MuonResiduals6DOFFitter() override {}

  int type() const override { return MuonResidualsFitter::k6DOF; }

  int npar() override {
    if (residualsModel() == kPureGaussian || residualsModel() == kPureGaussian2D || residualsModel() == kGaussPowerTails) return kNPar - 4;
    else if (residualsModel() == kPowerLawTails) return kNPar;
    else if (residualsModel() == kROOTVoigt) return kNPar;
    else assert(false);
  }
  int ndata() override { return kNData; }

  double sumofweights() override;
  bool fit(Alignable *ali) override;
  double plot(std::string name, TFileDirectory *dir, Alignable *ali) override;

  void correctBField() override;

  TTree * readNtuple(std::string fname, unsigned int wheel, unsigned int station, unsigned int sector, unsigned int preselected = 1);

protected:
  void inform(TMinuit *tMinuit) override;
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonResiduals6DOFFitter_H
