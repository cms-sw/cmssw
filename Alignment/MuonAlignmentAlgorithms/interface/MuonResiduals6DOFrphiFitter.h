#ifndef Alignment_MuonAlignmentAlgorithms_MuonResiduals6DOFrphiFitter_H
#define Alignment_MuonAlignmentAlgorithms_MuonResiduals6DOFrphiFitter_H

/** \class MuonResiduals6DOFrphiFitter
 *  $Date: Thu Apr 16 21:29:15 CDT 2009
 *  $Revision: 1.5 $ 
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#ifdef STANDALONE_FITTER
#include "MuonResidualsFitter.h"
#else
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#endif

class TTree;

class MuonResiduals6DOFrphiFitter : public MuonResidualsFitter {
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

  enum { kResid = 0, kResSlope, kPositionX, kPositionY, kAngleX, kAngleY, kRedChi2, kPz, kPt, kCharge, kNData };

  MuonResiduals6DOFrphiFitter(int residualsModel, int minHits, int useResiduals, bool weightAlignment = true)
      : MuonResidualsFitter(residualsModel, minHits, useResiduals, weightAlignment) {}

#ifndef STANDALONE_FITTER
  MuonResiduals6DOFrphiFitter(
      int residualsModel, int minHits, int useResiduals, const CSCGeometry *cscGeometry, bool weightAlignment = true)
      : MuonResidualsFitter(residualsModel, minHits, useResiduals, weightAlignment) {}
#endif

  ~MuonResiduals6DOFrphiFitter() override {}

  int type() const override { return MuonResidualsFitter::k6DOFrphi; }

  int npar() override {
    if (residualsModel() == kPureGaussian || residualsModel() == kPureGaussian2D ||
        residualsModel() == kGaussPowerTails)
      return kNPar - 2;
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

  void correctBField() override;

  TTree *readNtuple(std::string fname,
                    unsigned int endcap,
                    unsigned int station,
                    unsigned int ring,
                    unsigned int chamber,
                    unsigned int preselected = 1);

protected:
  void inform(TMinuit *tMinuit) override;

private:
  //const CSCGeometry *m_cscGeometry;
};

#endif  // Alignment_MuonAlignmentAlgorithms_MuonResiduals6DOFrphiFitter_H
