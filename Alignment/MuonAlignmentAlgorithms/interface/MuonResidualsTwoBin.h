#ifndef Alignment_MuonAlignmentAlgorithms_MuonResidualsTwoBin_H
#define Alignment_MuonAlignmentAlgorithms_MuonResidualsTwoBin_H

/** \class MuonResidualsTwoBin
 *  $Date: 2011/10/12 23:45:21 $
 *  $Revision: 1.11 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "TMath.h"

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
  int type() const { assert(m_pos->type() == m_neg->type());  return m_pos->type(); };
  int useRes() const { return m_pos->useRes(); };

  void fix(int parNum, bool value=true) {
    m_pos->fix(parNum, value);
    m_neg->fix(parNum, value);
  };

  bool fixed(int parNum) {
    return m_pos->fixed(parNum)  &&  m_neg->fixed(parNum);
  };

  void setPrintLevel(int printLevel) const {
    m_pos->setPrintLevel(printLevel);
    m_neg->setPrintLevel(printLevel);
  }

  void setStrategy(int strategy) const {
    m_pos->setStrategy(strategy);
    m_neg->setStrategy(strategy);
  }

  void fill(char charge, double *residual) {
    if (!m_twoBin  ||  charge > 0) m_pos->fill(residual);
    else m_neg->fill(residual);
  };

  bool fit(Alignable *ali) {
    return (m_twoBin ? (m_pos->fit(ali)  &&  m_neg->fit(ali)) : m_pos->fit(ali));
  };
  double value(int parNum) {
    return (m_twoBin ? ((m_pos->value(parNum) + m_neg->value(parNum)) / 2.) : m_pos->value(parNum));
  };
  double errorerror(int parNum) {
    return (m_twoBin ? (sqrt(pow(m_pos->errorerror(parNum), 2.) + pow(m_neg->errorerror(parNum), 2.)) / 2.) : m_pos->errorerror(parNum));
  };
  double antisym(int parNum) {
    return (m_twoBin ? ((m_pos->value(parNum) - m_neg->value(parNum)) / 2.) : 0.);
  };
  double loglikelihood() {
    return (m_twoBin ? (m_pos->loglikelihood() + m_neg->loglikelihood()) : m_pos->loglikelihood());
  };
  double numsegments() {
    return (m_twoBin ? (m_pos->numsegments() + m_neg->numsegments()) : m_pos->numsegments());
  };
  double sumofweights() {
    return (m_twoBin ? (m_pos->sumofweights() + m_neg->sumofweights()) : m_pos->sumofweights());
  };

  // demonstration plots
  double plot(std::string name, TFileDirectory *dir, Alignable *ali) {
    if (m_twoBin) {
      std::string namePos = name + std::string("Pos");
      std::string nameNeg = name + std::string("Neg");
      double output = 0.;
      output += m_pos->plot(namePos, dir, ali);
      output += m_neg->plot(nameNeg, dir, ali);
      return output;
    }
    else {
      return m_pos->plot(name, dir, ali);
    }
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

  double median(int which) {
    std::vector<double> residuals;
    for (std::vector<double*>::const_iterator r = residualsPos_begin();  r != residualsPos_end();  ++r) {
      residuals.push_back((*r)[which]);
    }
    if (m_twoBin) {
      for (std::vector<double*>::const_iterator r = residualsNeg_begin();  r != residualsNeg_end();  ++r) {
	residuals.push_back((*r)[which]);
      }
    }
    std::sort(residuals.begin(), residuals.end());
    int length = residuals.size();
    return residuals[length/2];
  };

  double mean(int which, double truncate) {
    double sum = 0.;
    double n = 0.;
    for (std::vector<double*>::const_iterator r = residualsPos_begin();  r != residualsPos_end();  ++r) {
      double value = (*r)[which];
      if (fabs(value) < truncate) {
	sum += value;
	n += 1.;
      }
    }
    if (m_twoBin) {
      for (std::vector<double*>::const_iterator r = residualsNeg_begin();  r != residualsNeg_end();  ++r) {
	double value = (*r)[which];
	if (fabs(value) < truncate) {
	  sum += value;
	  n += 1.;
	}
      }
    }
    return sum/n;
  };

  double wmean(int which, int whichredchi2, double truncate) {
    double sum = 0.;
    double n = 0.;
    for (std::vector<double*>::const_iterator r = residualsPos_begin();  r != residualsPos_end();  ++r) {
      double value = (*r)[which];
      if (fabs(value) < truncate) {
	double weight = 1./(*r)[whichredchi2];
	if (TMath::Prob(1./weight*12, 12) < 0.99) {
	  sum += weight*value;
	  n += weight;
	}
      }
    }
    if (m_twoBin) {
      for (std::vector<double*>::const_iterator r = residualsNeg_begin();  r != residualsNeg_end();  ++r) {
	double value = (*r)[which];
	if (fabs(value) < truncate) {
	  double weight = 1./(*r)[whichredchi2];
	  if (TMath::Prob(1./weight*12, 12) < 0.99) {
	    sum += weight*value;
	    n += weight;
	  }
	}
      }
    }
    return sum/n;
  };

  double stdev(int which, double truncate) {
    double sum2 = 0.;
    double sum = 0.;
    double n = 0.;
    for (std::vector<double*>::const_iterator r = residualsPos_begin();  r != residualsPos_end();  ++r) {
      double value = (*r)[which];
      if (fabs(value) < truncate) {
	sum2 += value*value;
	sum += value;
	n += 1.;
      }
    }
    if (m_twoBin) {
      for (std::vector<double*>::const_iterator r = residualsNeg_begin();  r != residualsNeg_end();  ++r) {
	double value = (*r)[which];
	if (fabs(value) < truncate) {
	  sum2 += value*value;
	  sum += value;
	  n += 1.;
	}
      }
    }
    return sqrt(sum2/n - pow(sum/n, 2));
  };

  void plotsimple(std::string name, TFileDirectory *dir, int which, double multiplier) {
    if (m_twoBin) {
      std::string namePos = name + std::string("Pos");
      std::string nameNeg = name + std::string("Neg");
      m_pos->plotsimple(namePos, dir, which, multiplier);
      m_neg->plotsimple(nameNeg, dir, which, multiplier);
    }
    else {
      m_pos->plotsimple(name, dir, which, multiplier);
    }
  };

  void plotweighted(std::string name, TFileDirectory *dir, int which, int whichredchi2, double multiplier) {
    if (m_twoBin) {
      std::string namePos = name + std::string("Pos");
      std::string nameNeg = name + std::string("Neg");
      m_pos->plotweighted(namePos, dir, which, whichredchi2, multiplier);
      m_neg->plotweighted(nameNeg, dir, which, whichredchi2, multiplier);
    }
    else {
      m_pos->plotweighted(name, dir, which, whichredchi2, multiplier);
    }
  };

  void selectPeakResiduals(double nsigma, int nvar, int *vars)
  {
    if (m_twoBin) {
      m_pos->selectPeakResiduals(nsigma, nvar, vars);
      m_neg->selectPeakResiduals(nsigma, nvar, vars);
    }
    else {
      m_pos->selectPeakResiduals(nsigma, nvar, vars);
    }
  }
  
  void correctBField()
  {
    m_pos->correctBField();
    //if (m_twoBin) m_neg->correctBField();
  };

  void eraseNotSelectedResiduals()
  {
    if (m_twoBin) {
      m_pos->eraseNotSelectedResiduals();
      m_neg->eraseNotSelectedResiduals();
    }
    else {
      m_pos->eraseNotSelectedResiduals();
    }
  }

  std::vector<double*>::const_iterator residualsPos_begin() const { return m_pos->residuals_begin(); };
  std::vector<double*>::const_iterator residualsPos_end() const { return m_pos->residuals_end(); };
  std::vector<double*>::const_iterator residualsNeg_begin() const { return m_neg->residuals_begin(); };
  std::vector<double*>::const_iterator residualsNeg_end() const { return m_neg->residuals_end(); };

  std::vector<bool>::const_iterator residualsPos_ok_begin() const { return m_pos->selectedResidualsFlags().begin(); };
  std::vector<bool>::const_iterator residualsPos_ok_end() const { return m_pos->selectedResidualsFlags().end(); };
  std::vector<bool>::const_iterator residualsNeg_ok_begin() const { return m_neg->selectedResidualsFlags().begin(); };
  std::vector<bool>::const_iterator residualsNeg_ok_end() const { return m_neg->selectedResidualsFlags().end(); };

protected:
  bool m_twoBin;
  MuonResidualsFitter *m_pos, *m_neg;
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonResidualsTwoBin_H
