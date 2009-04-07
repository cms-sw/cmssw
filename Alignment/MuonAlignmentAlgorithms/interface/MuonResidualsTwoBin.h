#ifndef Alignment_MuonAlignmentAlgorithms_MuonResidualsTwoBin_H
#define Alignment_MuonAlignmentAlgorithms_MuonResidualsTwoBin_H

/** \class MuonResidualsTwoBin
 *  $Date: 2009/03/24 17:17:43 $
 *  $Revision: 1.3 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h"

class MuonResidualsTwoBin {
public:
  MuonResidualsTwoBin(bool twoBin, MuonResidualsFitter *pos, MuonResidualsFitter *neg): m_twoBin(twoBin), m_pos(pos), m_neg(neg) {};
  ~MuonResidualsTwoBin() {
    if (m_pos != NULL) delete m_pos;
    if (m_neg != NULL) delete m_neg;
  };

  int residualsModel() const { assert(m_pos->residualsModel() == m_neg->residualsModel());  return m_pos->residualsModel(); };
  long numResidualsPos() const { return m_pos->numResiduals(); };
  long numResidualsNeg() const { return m_neg->numResiduals(); };
  int npar() { assert(m_pos->npar() == m_neg->npar());  return m_pos->npar(); };
  int ndata() { assert(m_pos->ndata() == m_neg->ndata());  return m_pos->ndata(); };

  void fix(int parNum, bool value=true) {
    m_pos->fix(parNum, value);
    m_neg->fix(parNum, value);
  };

  bool fixed(int parNum) {
    return m_pos->fixed(parNum)  &&  m_neg->fixed(parNum);
  };

  void fill(char charge, double *residual) {
    if (!m_twoBin  ||  charge > 0) m_pos->fill(residual);
    else m_neg->fill(residual);
  };

  bool fit(double v1) {
    if (m_twoBin) return m_pos->fit(v1)  &&  m_neg->fit(v1);
    else return m_pos->fit(v1);
  };
  double value(int parNum) {
    if (m_twoBin) return (m_pos->value(parNum) + m_neg->value(parNum)) / 2.;
    else return m_pos->value(parNum);
  };
  double error(int parNum) {
    if (m_twoBin) return sqrt(pow(m_pos->error(parNum), 2.) + pow(m_neg->error(parNum), 2.)) / 2.;
    else return m_pos->error(parNum);
  };
  double antisym(int parNum) {
    if (m_twoBin) return (m_pos->value(parNum) - m_neg->value(parNum)) / 2.;
    else return 0.;
  };

  // demonstration plots
  void plot(double v1, std::string name, TFileDirectory *dir) {
    if (m_twoBin) {
      std::string namePos = name + std::string("Pos");
      std::string nameNeg = name + std::string("Neg");
      m_pos->plot(v1, namePos, dir);
      m_neg->plot(v1, nameNeg, dir);
    }
    else {
      m_pos->plot(v1, name, dir);
    }
  };
  double redchi2(double v1, std::string name, TFileDirectory *dir=NULL, bool write=false, int bins=100, double low=-5., double high=5.) {
    if (m_twoBin) {
      std::string namePos = name + std::string("Pos");
      std::string nameNeg = name + std::string("Neg");
      double chi2 = 0.;
      chi2 += m_pos->redchi2(v1, namePos, dir, write, bins, low, high);
      chi2 += m_neg->redchi2(v1, nameNeg, dir, write, bins, low, high);
      return chi2;
    }
    else return m_pos->redchi2(v1, name, dir, write, bins, low, high);
  };

  // I/O of temporary files for collect mode
  void write(FILE *file, int which=0) {
    if (m_twoBin) {
      m_pos->write(file, 2*which);
      m_neg->write(file, 2*which + 1);
    }
    else {
      m_pos->write(file, which);
    }
  };
  void read(FILE *file, int which=0) {
    if (m_twoBin) {
      m_pos->read(file, 2*which);
      m_neg->read(file, 2*which + 1);
    }
    else {
      m_pos->read(file, which);
    }
  };

  std::vector<double*>::const_iterator residualsPos_begin() const { return m_pos->residuals_begin(); };
  std::vector<double*>::const_iterator residualsPos_end() const { return m_pos->residuals_end(); };
  std::vector<double*>::const_iterator residualsNeg_begin() const { return m_neg->residuals_begin(); };
  std::vector<double*>::const_iterator residualsNeg_end() const { return m_neg->residuals_end(); };

protected:
  bool m_twoBin;
  MuonResidualsFitter *m_pos, *m_neg;
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonResidualsTwoBin_H
