#include <iostream>

#include "CLHEP/Matrix/Matrix.h"
#include "TMatrixDEigen.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/MuonAlignmentAlgorithms/plugins/CSCChamberFitter.h"

const double infinity =
    0.1;  // this is huge because all alignments are angles in radians; but we need a not-too-large value for numerical stability
          // should become a parameter someday

CSCChamberFitter::CSCChamberFitter(const edm::ParameterSet &iConfig,
                                   std::vector<CSCPairResidualsConstraint *> &residualsConstraints) {
  m_name = iConfig.getParameter<std::string>("name");
  m_alignables = iConfig.getParameter<std::vector<std::string> >("alignables");
  if (m_alignables.empty()) {
    throw cms::Exception("BadConfig") << "Fitter " << m_name << " has no alignables!" << std::endl;
  }

  int i = 0;
  for (const auto &m_alignable : m_alignables) {
    if (alignableId(m_alignable) == -1)
      m_frames.push_back(i);
    i++;
  }

  m_fixed = -1;
  std::string fixed = iConfig.getParameter<std::string>("fixed");
  if (!fixed.empty()) {
    int i = 0;
    for (const auto &m_alignable : m_alignables) {
      if (fixed == m_alignable) {
        m_fixed = i;
      }
      i++;
    }
    if (m_fixed == -1)
      throw cms::Exception("BadConfig") << "Cannot fix unrecognized alignable " << fixed << std::endl;
  }

  int numConstraints = 0;
  std::vector<edm::ParameterSet> constraints = iConfig.getParameter<std::vector<edm::ParameterSet> >("constraints");
  for (const auto &constraint : constraints) {
    int i = index(constraint.getParameter<std::string>("i"));
    int j = index(constraint.getParameter<std::string>("j"));
    double value = constraint.getParameter<double>("value");
    double error = constraint.getParameter<double>("error");

    if (i < 0)
      throw cms::Exception("BadConfig") << "Unrecognized alignable " << constraint.getParameter<std::string>("i")
                                        << " in constraint " << numConstraints << " of fitter " << m_name << std::endl;
    if (j < 0)
      throw cms::Exception("BadConfig") << "Unrecognized alignable " << constraint.getParameter<std::string>("j")
                                        << " in constraint " << numConstraints << " of fitter " << m_name << std::endl;
    if (error <= 0.)
      throw cms::Exception("BadConfig") << "Non-positive uncertainty in constraint " << numConstraints << " of fitter "
                                        << m_name << std::endl;
    if (i == j)
      throw cms::Exception("BadConfig") << "Self-connection from " << constraint.getParameter<std::string>("i")
                                        << " to " << constraint.getParameter<std::string>("j")
                                        << " is not allowed in constraint " << numConstraints << " of fitter " << m_name
                                        << std::endl;

    m_constraints.push_back(new CSCPairConstraint(i, j, value, error));
    numConstraints++;
  }

  // insert CSCPairResidualsConstraints
  for (unsigned int i = 0; i < m_alignables.size(); i++) {
    std::string alignable_i = m_alignables[i];
    long id_i = alignableId(alignable_i);
    if (id_i != -1) {
      CSCDetId cscid_i(id_i);

      for (unsigned int j = 0; j < m_alignables.size(); j++) {
        std::string alignable_j = m_alignables[j];
        long id_j = alignableId(alignable_j);
        if (i != j && id_j != -1) {
          CSCDetId cscid_j(id_j);

          if (!(cscid_i.station() == 1 && cscid_i.ring() == 3 && cscid_j.station() == 1 && cscid_j.ring() == 3)) {
            int next_chamber = cscid_i.chamber() + 1;
            if (cscid_i.station() > 1 && cscid_i.ring() == 1 && next_chamber == 19)
              next_chamber = 1;
            else if (!(cscid_i.station() > 1 && cscid_i.ring() == 1) && next_chamber == 37)
              next_chamber = 1;
            if (cscid_i.endcap() == cscid_j.endcap() && cscid_i.station() == cscid_j.station() &&
                cscid_i.ring() == cscid_j.ring() && next_chamber == cscid_j.chamber()) {
              CSCPairResidualsConstraint *residualsConstraint =
                  new CSCPairResidualsConstraint(residualsConstraints.size(), i, j, cscid_i, cscid_j);
              m_constraints.push_back(residualsConstraint);
              residualsConstraints.push_back(residualsConstraint);
              numConstraints++;
            }
          }
        }
      }
    }
  }

  std::map<int, bool> touched;
  for (unsigned int i = 0; i < m_alignables.size(); i++)
    touched[i] = false;
  walk(touched, 0);
  for (unsigned int i = 0; i < m_alignables.size(); i++) {
    if (!touched[i])
      throw cms::Exception("BadConfig") << "Fitter " << m_name << " is not a connected graph (no way to get to "
                                        << m_alignables[i] << " from " << m_alignables[0] << ", for instance)"
                                        << std::endl;
  }
}

int CSCChamberFitter::index(std::string alignable) const {
  int i = 0;
  for (const auto &m_alignable : m_alignables) {
    if (m_alignable == alignable)
      return i;
    i++;
  }
  return -1;
}

void CSCChamberFitter::walk(std::map<int, bool> &touched, int alignable) const {
  touched[alignable] = true;

  for (auto m_constraint : m_constraints) {
    if (alignable == m_constraint->i() || alignable == m_constraint->j()) {
      if (!touched[m_constraint->i()])
        walk(touched, m_constraint->i());
      if (!touched[m_constraint->j()])
        walk(touched, m_constraint->j());
    }
  }
}

long CSCChamberFitter::alignableId(std::string alignable) const {
  if (alignable.size() != 9)
    return -1;

  if (alignable[0] == 'M' && alignable[1] == 'E') {
    int endcap = -1;
    if (alignable[2] == '+')
      endcap = 1;
    else if (alignable[2] == '-')
      endcap = 2;

    if (endcap != -1) {
      int station = -1;
      if (alignable[3] == '1')
        station = 1;
      else if (alignable[3] == '2')
        station = 2;
      else if (alignable[3] == '3')
        station = 3;
      else if (alignable[3] == '4')
        station = 4;

      if (alignable[4] == '/' && station != -1) {
        int ring = -1;
        if (alignable[5] == '1')
          ring = 1;
        else if (alignable[5] == '2')
          ring = 2;
        else if (alignable[5] == '3')
          ring = 3;
        else if (alignable[5] == '4')
          ring = 4;
        if (station > 1 && ring > 2)
          return -1;

        if (alignable[6] == '/' && ring != -1) {
          int chamber = -1;
          if (alignable[7] == '0' && alignable[8] == '1')
            chamber = 1;
          else if (alignable[7] == '0' && alignable[8] == '2')
            chamber = 2;
          else if (alignable[7] == '0' && alignable[8] == '3')
            chamber = 3;
          else if (alignable[7] == '0' && alignable[8] == '4')
            chamber = 4;
          else if (alignable[7] == '0' && alignable[8] == '5')
            chamber = 5;
          else if (alignable[7] == '0' && alignable[8] == '6')
            chamber = 6;
          else if (alignable[7] == '0' && alignable[8] == '7')
            chamber = 7;
          else if (alignable[7] == '0' && alignable[8] == '8')
            chamber = 8;
          else if (alignable[7] == '0' && alignable[8] == '9')
            chamber = 9;
          else if (alignable[7] == '1' && alignable[8] == '0')
            chamber = 10;
          else if (alignable[7] == '1' && alignable[8] == '1')
            chamber = 11;
          else if (alignable[7] == '1' && alignable[8] == '2')
            chamber = 12;
          else if (alignable[7] == '1' && alignable[8] == '3')
            chamber = 13;
          else if (alignable[7] == '1' && alignable[8] == '4')
            chamber = 14;
          else if (alignable[7] == '1' && alignable[8] == '5')
            chamber = 15;
          else if (alignable[7] == '1' && alignable[8] == '6')
            chamber = 16;
          else if (alignable[7] == '1' && alignable[8] == '7')
            chamber = 17;
          else if (alignable[7] == '1' && alignable[8] == '8')
            chamber = 18;
          else if (alignable[7] == '1' && alignable[8] == '9')
            chamber = 19;
          else if (alignable[7] == '2' && alignable[8] == '0')
            chamber = 20;
          else if (alignable[7] == '2' && alignable[8] == '1')
            chamber = 21;
          else if (alignable[7] == '2' && alignable[8] == '2')
            chamber = 22;
          else if (alignable[7] == '2' && alignable[8] == '3')
            chamber = 23;
          else if (alignable[7] == '2' && alignable[8] == '4')
            chamber = 24;
          else if (alignable[7] == '2' && alignable[8] == '5')
            chamber = 25;
          else if (alignable[7] == '2' && alignable[8] == '6')
            chamber = 26;
          else if (alignable[7] == '2' && alignable[8] == '7')
            chamber = 27;
          else if (alignable[7] == '2' && alignable[8] == '8')
            chamber = 28;
          else if (alignable[7] == '2' && alignable[8] == '9')
            chamber = 29;
          else if (alignable[7] == '3' && alignable[8] == '0')
            chamber = 30;
          else if (alignable[7] == '3' && alignable[8] == '1')
            chamber = 31;
          else if (alignable[7] == '3' && alignable[8] == '2')
            chamber = 32;
          else if (alignable[7] == '3' && alignable[8] == '3')
            chamber = 33;
          else if (alignable[7] == '3' && alignable[8] == '4')
            chamber = 34;
          else if (alignable[7] == '3' && alignable[8] == '5')
            chamber = 35;
          else if (alignable[7] == '3' && alignable[8] == '6')
            chamber = 36;

          if (station > 1 && ring == 1 && chamber > 18)
            return -1;

          if (chamber != -1) {
            return CSCDetId(endcap, station, ring, chamber, 0).rawId();
          }
        }
      }
    }
  }

  return -1;
}

bool CSCChamberFitter::isFrame(int i) const {
  for (int m_frame : m_frames) {
    if (i == m_frame)
      return true;
  }
  return false;
}

double CSCChamberFitter::chi2(const AlgebraicVector &A, double lambda) const {
  double sumFixed = 0.;

  if (m_fixed == -1) {
    for (unsigned int i = 0; i < m_alignables.size(); i++) {
      if (!isFrame(i)) {
        sumFixed += A[i];
      }
    }
  } else {
    sumFixed = A[m_fixed];
  }

  double s = lambda * sumFixed * sumFixed;
  for (auto m_constraint : m_constraints) {
    if (m_constraint->valid()) {
      s += pow(m_constraint->value() - A[m_constraint->i()] + A[m_constraint->j()], 2) / m_constraint->error() /
           m_constraint->error();
    }
  }
  return s;
}

double CSCChamberFitter::lhsVector(int k) const {
  double s = 0.;
  for (auto m_constraint : m_constraints) {
    if (m_constraint->valid()) {
      double d = 2. * m_constraint->value() / m_constraint->error() / m_constraint->error();
      if (m_constraint->i() == k)
        s += d;
      if (m_constraint->j() == k)
        s -= d;
    }
  }
  return s;
}

double CSCChamberFitter::hessian(int k, int l, double lambda) const {
  double s = 0.;

  if (m_fixed == -1) {
    if (!isFrame(k) && !isFrame(l))
      s += 2. * lambda;
  } else {
    if (k == l && l == m_fixed)
      s += 2. * lambda;
  }

  for (auto m_constraint : m_constraints) {
    double d = 2. / infinity / infinity;
    if (m_constraint->valid()) {
      d = 2. / m_constraint->error() / m_constraint->error();
    }

    if (k == l && (m_constraint->i() == k || m_constraint->j() == k))
      s += d;
    if ((m_constraint->i() == k && m_constraint->j() == l) || (m_constraint->j() == k && m_constraint->i() == l))
      s -= d;
  }
  return s;
}

bool CSCChamberFitter::fit(std::vector<CSCAlignmentCorrections *> &corrections) const {
  double lambda = 1. / infinity / infinity;

  AlgebraicVector A(m_alignables.size());
  AlgebraicVector V(m_alignables.size());
  AlgebraicMatrix M(m_alignables.size(), m_alignables.size());

  for (unsigned int k = 0; k < m_alignables.size(); k++) {
    A[k] = 0.;
    V[k] = lhsVector(k);
    for (unsigned int l = 0; l < m_alignables.size(); l++) {
      M[k][l] = hessian(k, l, lambda);
    }
  }

  double oldchi2 = chi2(A, lambda);

  int ierr;
  M.invert(ierr);
  if (ierr != 0) {
    edm::LogError("CSCOverlapsAlignmentAlgorithm")
        << "Matrix inversion failed for fitter " << m_name << " matrix is " << M << std::endl;
    return false;
  }

  A = M * V;  // that's the alignment step

  ///// everything else is for reporting
  CSCAlignmentCorrections *correction = new CSCAlignmentCorrections(m_name, oldchi2, chi2(A, lambda));

  for (unsigned int i = 0; i < m_alignables.size(); i++) {
    if (!isFrame(i)) {
      correction->insertCorrection(m_alignables[i], CSCDetId(alignableId(m_alignables[i])), A[i]);
    }
  }

  // we have to switch to a completely different linear algebrea
  // package because CLHEP doesn't compute
  // eigenvectors/diagonalization (?!?)
  TMatrixD tmatrix(m_alignables.size(), m_alignables.size());
  for (unsigned int i = 0; i < m_alignables.size(); i++) {
    for (unsigned int j = 0; j < m_alignables.size(); j++) {
      tmatrix[i][j] = M[i][j];
    }
  }
  TMatrixDEigen tmatrixdeigen(tmatrix);
  const TMatrixD &basis = tmatrixdeigen.GetEigenVectors();
  TMatrixD invbasis = tmatrixdeigen.GetEigenVectors();
  invbasis.Invert();
  TMatrixD diagonalized = invbasis * (tmatrix * basis);

  for (unsigned int i = 0; i < m_alignables.size(); i++) {
    std::vector<double> coefficient;
    std::vector<std::string> modename;
    std::vector<long> modeid;
    for (unsigned int j = 0; j < m_alignables.size(); j++) {
      coefficient.push_back(invbasis[i][j]);
      modename.push_back(m_alignables[j]);
      modeid.push_back(alignableId(m_alignables[j]));
    }

    correction->insertMode(
        coefficient, modename, modeid, sqrt(2. * fabs(diagonalized[i][i])) * (diagonalized[i][i] >= 0. ? 1. : -1.));
  }

  for (auto m_constraint : m_constraints) {
    if (m_constraint->valid()) {
      double residual = m_constraint->value() - A[m_constraint->i()] + A[m_constraint->j()];
      correction->insertResidual(m_alignables[m_constraint->i()],
                                 m_alignables[m_constraint->j()],
                                 m_constraint->value(),
                                 m_constraint->error(),
                                 residual,
                                 residual / m_constraint->error());
    }
  }

  corrections.push_back(correction);
  return true;
}

void CSCChamberFitter::radiusCorrection(AlignableNavigator *alignableNavigator,
                                        AlignmentParameterStore *alignmentParameterStore,
                                        bool combineME11) const {
  double sum_phipos_residuals = 0.;
  double num_valid = 0.;
  double sum_radius = 0.;
  double num_total = 0.;
  for (auto m_constraint : m_constraints) {
    CSCPairResidualsConstraint *residualsConstraint = dynamic_cast<CSCPairResidualsConstraint *>(m_constraint);
    if (residualsConstraint != nullptr) {
      if (residualsConstraint->valid()) {
        sum_phipos_residuals += residualsConstraint->value();
        num_valid += 1.;
      }

      sum_radius += residualsConstraint->radius(true);
      num_total += 1.;
    }
  }
  if (num_valid == 0. || num_total == 0.)
    return;
  double average_phi_residual = sum_phipos_residuals / num_valid;
  double average_radius = sum_radius / num_total;

  double radial_correction = average_phi_residual * average_radius * num_total / (2. * M_PI);

  for (auto m_constraint : m_constraints) {
    CSCPairResidualsConstraint *residualsConstraint = dynamic_cast<CSCPairResidualsConstraint *>(m_constraint);
    if (residualsConstraint != nullptr) {
      const DetId id(residualsConstraint->id_i());
      Alignable *alignable = alignableNavigator->alignableFromDetId(id).alignable();
      Alignable *also = nullptr;
      if (combineME11 && residualsConstraint->id_i().station() == 1 && residualsConstraint->id_i().ring() == 1) {
        CSCDetId alsoid(residualsConstraint->id_i().endcap(), 1, 4, residualsConstraint->id_i().chamber(), 0);
        const DetId alsoid2(alsoid);
        also = alignableNavigator->alignableFromDetId(alsoid2).alignable();
      }

      AlgebraicVector params(6);
      AlgebraicSymMatrix cov(6);

      params[1] = radial_correction;
      cov[1][1] = 1e-6;

      AlignmentParameters *parnew = alignable->alignmentParameters()->cloneFromSelected(params, cov);
      alignable->setAlignmentParameters(parnew);
      alignmentParameterStore->applyParameters(alignable);
      alignable->alignmentParameters()->setValid(true);
      if (also != nullptr) {
        AlignmentParameters *parnew2 = also->alignmentParameters()->cloneFromSelected(params, cov);
        also->setAlignmentParameters(parnew2);
        alignmentParameterStore->applyParameters(also);
        also->alignmentParameters()->setValid(true);
      }
    }
  }
}
