#ifndef Alignment_MuonAlignmentAlgorithms_MuonResidualsTwoBin_H
#define Alignment_MuonAlignmentAlgorithms_MuonResidualsTwoBin_H

/** \class MuonResidualsTwoBin
 *  $Date: Mon Mar 23 18:32:20 CDT 2009 $
 *  $Revision: 1.0 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h"

class MuonResidualsTwoBin {
public:
  MuonResidualsTwoBin(MuonResidualsFitter *pos, MuonResidualsFitter *neg): m_pos(pos), m_neg(neg) {};
  ~MuonResidualsTwoBin() { delete m_pos;  delete m_neg; };

  void fix(int parNum, bool value=true) {
    m_pos->fix(parNum, value);
    m_neg->fix(parNum, value);
  };

  bool fixed(int parNum) {
    return m_pos->fixed(parNum)  &&  m_neg->fixed(parNum);
  };

  void fill(char charge, double *residual) {
    if (charge > 0) m_pos->fill(residual);
    else m_neg->fill(residual);
  };

  bool fit(double v1) { return m_pos->fit(v1)  &&  m_neg->fit(v1); };
  double value(int parNum) { return (m_pos->value(parNum) + m_neg->value(parNum)) / 2.; };
  double antisym(int parNum) { return (m_pos->value(parNum) - m_neg->value(parNum)) / 2.; };
  double error(int parNum) { return sqrt(pow(m_pos->error(parNum), 2.) + pow(m_neg->error(parNum), 2.)) / 2.; };
  double uperr(int parNum) { return sqrt(pow(m_pos->uperr(parNum), 2.) + pow(m_neg->uperr(parNum), 2.)) / 2.; };
  double downerr(int parNum) { return -sqrt(pow(m_pos->downerr(parNum), 2.) + pow(m_neg->downerr(parNum), 2.)) / 2.; };
  double minoserr(int parNum) { return sqrt(pow(m_pos->minoserr(parNum), 2.) + pow(m_neg->minoserr(parNum), 2.)) / 2.; };

  // demonstration plots
  void plot(double v1, std::string name, TFileDirectory *dir) {
    std::string namePos = name + std::string("Pos");
    std::string nameNeg = name + std::string("Neg");
    m_pos->plot(v1, namePos, dir);
    m_neg->plot(v1, nameNeg, dir);
  };
  double redchi2(double v1, std::string name, TFileDirectory *dir=NULL, bool write=false, int bins=100, double low=-5., double high=5.) {
    std::string namePos = name + std::string("Pos");
    std::string nameNeg = name + std::string("Neg");
    double chi2 = 0.;
    chi2 += m_pos->redchi2(v1, namePos, dir, write, bins, low, high);
    chi2 += m_neg->redchi2(v1, namePos, dir, write, bins, low, high);
    return chi2;
  };

  // I/O of temporary files for collect mode
  void write(FILE *file, int which=0) { m_pos->write(file, 2*which);  m_neg->write(file, 2*which + 1); };
  void read(FILE *file, int which=0) { m_pos->read(file, 2*which);  m_neg->read(file, 2*which + 1); }

protected:
  MuonResidualsFitter *m_pos, *m_neg;
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonResidualsTwoBin_H
