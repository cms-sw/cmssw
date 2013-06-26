#ifndef Alignment_MuonAlignmentAlgorithms_MuonResiduals5DOFFitter_H
#define Alignment_MuonAlignmentAlgorithms_MuonResiduals5DOFFitter_H

/** \class MuonResiduals5DOFFitter
 *  $Date: Fri Apr 17 15:29:54 CDT 2009
 *  $Revision: 1.6 $ 
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#ifdef STANDALONE_FITTER
#include "MuonResidualsFitter.h"
#else
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h"
#endif

class TTree;

class MuonResiduals5DOFFitter: public MuonResidualsFitter
{
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
    kPz,
    kPt,
    kCharge,
    kNData
  };

  MuonResiduals5DOFFitter(int residualsModel, int minHits, int useResiduals, bool weightAlignment=true): MuonResidualsFitter(residualsModel, minHits, useResiduals, weightAlignment) {}
  virtual ~MuonResiduals5DOFFitter(){}

  int type() const { return MuonResidualsFitter::k5DOF; }

  int npar() {
    if (residualsModel() == kPureGaussian || residualsModel() == kPureGaussian2D || residualsModel() == kGaussPowerTails) return kNPar - 2;
    else if (residualsModel() == kPowerLawTails) return kNPar;
    else if (residualsModel() == kROOTVoigt) return kNPar;
    else assert(false);
  }
  int ndata() { return kNData; }

  double sumofweights();
  bool fit(Alignable *ali);
  double plot(std::string name, TFileDirectory *dir, Alignable *ali);

  void correctBField();

  TTree * readNtuple(std::string fname, unsigned int wheel, unsigned int station, unsigned int sector, unsigned int preselected = 1);

protected:
  void inform(TMinuit *tMinuit);
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonResiduals5DOFFitter_H
