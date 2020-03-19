#include "PhysicsTools/CandUtils/interface/EventShapeVariables.h"

#include "TMath.h"

/// constructor from reco::Candidates
EventShapeVariables::EventShapeVariables(const edm::View<reco::Candidate>& inputVectors) : eigenVectors_(3, 3) {
  inputVectors_.reserve(inputVectors.size());
  for (const auto& vec : inputVectors) {
    inputVectors_.push_back(math::XYZVector(vec.px(), vec.py(), vec.pz()));
  }
  //default values
  set_r(2.);
  setFWmax(10);
}

/// constructor from XYZ coordinates
EventShapeVariables::EventShapeVariables(const std::vector<math::XYZVector>& inputVectors)
    : inputVectors_(inputVectors), eigenVectors_(3, 3) {
  //default values
  set_r(2.);
  setFWmax(10);
}

/// constructor from rho eta phi coordinates
EventShapeVariables::EventShapeVariables(const std::vector<math::RhoEtaPhiVector>& inputVectors) : eigenVectors_(3, 3) {
  inputVectors_.reserve(inputVectors.size());
  for (const auto& vec : inputVectors) {
    inputVectors_.push_back(math::XYZVector(vec.x(), vec.y(), vec.z()));
  }
  //default values
  set_r(2.);
  setFWmax(10);
}

/// constructor from r theta phi coordinates
EventShapeVariables::EventShapeVariables(const std::vector<math::RThetaPhiVector>& inputVectors) : eigenVectors_(3, 3) {
  inputVectors_.reserve(inputVectors.size());
  for (const auto& vec : inputVectors) {
    inputVectors_.push_back(math::XYZVector(vec.x(), vec.y(), vec.z()));
  }
  //default values
  set_r(2.);
  setFWmax(10);
}

/// the return value is 1 for spherical events and 0 for events linear in r-phi. This function
/// needs the number of steps to determine how fine the granularity of the algorithm in phi
/// should be
double EventShapeVariables::isotropy(const unsigned int& numberOfSteps) const {
  const double deltaPhi = 2 * TMath::Pi() / numberOfSteps;
  double phi = 0, eIn = -1., eOut = -1.;
  for (unsigned int i = 0; i < numberOfSteps; ++i) {
    phi += deltaPhi;
    double sum = 0;
    double cosphi = TMath::Cos(phi);
    double sinphi = TMath::Sin(phi);
    for (const auto& vec : inputVectors_) {
      // sum over inner product of unit vectors and momenta
      sum += TMath::Abs(cosphi * vec.x() + sinphi * vec.y());
    }
    if (eOut < 0. || sum < eOut)
      eOut = sum;
    if (eIn < 0. || sum > eIn)
      eIn = sum;
  }
  return (eIn - eOut) / eIn;
}

/// the return value is 1 for spherical and 0 linear events in r-phi. This function needs the
/// number of steps to determine how fine the granularity of the algorithm in phi should be
double EventShapeVariables::circularity(const unsigned int& numberOfSteps) const {
  const double deltaPhi = 2 * TMath::Pi() / numberOfSteps;
  double circularity = -1, phi = 0, area = 0;
  for (const auto& vec : inputVectors_) {
    area += TMath::Sqrt(vec.x() * vec.x() + vec.y() * vec.y());
  }
  for (unsigned int i = 0; i < numberOfSteps; ++i) {
    phi += deltaPhi;
    double sum = 0, tmp = 0.;
    double cosphi = TMath::Cos(phi);
    double sinphi = TMath::Sin(phi);
    for (const auto& vec : inputVectors_) {
      sum += TMath::Abs(cosphi * vec.x() + sinphi * vec.y());
    }
    tmp = TMath::Pi() / 2 * sum / area;
    if (circularity < 0 || tmp < circularity) {
      circularity = tmp;
    }
  }
  return circularity;
}

/// set exponent for computation of momentum tensor and related products
void EventShapeVariables::set_r(double r) {
  r_ = r;
  /// invalidate previous cached computations
  tensors_computed_ = false;
  eigenValues_ = std::vector<double>(3, 0);
  eigenValuesNoNorm_ = std::vector<double>(3, 0);
}

/// helper function to fill the 3 dimensional momentum tensor from the inputVectors where needed
/// also fill the 3 dimensional vectors of eigen-values and eigen-vectors;
/// the largest (smallest) eigen-value is stored at index position 0 (2)
void EventShapeVariables::compTensorsAndVectors() {
  if (tensors_computed_)
    return;

  if (inputVectors_.size() < 2) {
    tensors_computed_ = true;
    return;
  }

  TMatrixDSym momentumTensor(3);
  momentumTensor.Zero();

  // fill momentumTensor from inputVectors
  double norm = 0.;
  for (const auto& vec : inputVectors_) {
    double p2 = vec.Dot(vec);
    double pR = (r_ == 2.) ? p2 : TMath::Power(p2, 0.5 * r_);
    norm += pR;
    double pRminus2 = (r_ == 2.) ? 1. : TMath::Power(p2, 0.5 * r_ - 1.);
    momentumTensor(0, 0) += pRminus2 * vec.x() * vec.x();
    momentumTensor(0, 1) += pRminus2 * vec.x() * vec.y();
    momentumTensor(0, 2) += pRminus2 * vec.x() * vec.z();
    momentumTensor(1, 0) += pRminus2 * vec.y() * vec.x();
    momentumTensor(1, 1) += pRminus2 * vec.y() * vec.y();
    momentumTensor(1, 2) += pRminus2 * vec.y() * vec.z();
    momentumTensor(2, 0) += pRminus2 * vec.z() * vec.x();
    momentumTensor(2, 1) += pRminus2 * vec.z() * vec.y();
    momentumTensor(2, 2) += pRminus2 * vec.z() * vec.z();
  }

  if (momentumTensor.IsSymmetric() && (momentumTensor.NonZeros() != 0)) {
    momentumTensor.EigenVectors(eigenValuesNoNormTmp_);
  }
  eigenValuesNoNorm_[0] = eigenValuesNoNormTmp_(0);
  eigenValuesNoNorm_[1] = eigenValuesNoNormTmp_(1);
  eigenValuesNoNorm_[2] = eigenValuesNoNormTmp_(2);

  // momentumTensor normalized to determinant 1
  momentumTensor *= (1. / norm);

  // now get eigens
  if (momentumTensor.IsSymmetric() && (momentumTensor.NonZeros() != 0)) {
    eigenVectors_ = momentumTensor.EigenVectors(eigenValuesTmp_);
  }
  eigenValues_[0] = eigenValuesTmp_(0);
  eigenValues_[1] = eigenValuesTmp_(1);
  eigenValues_[2] = eigenValuesTmp_(2);

  tensors_computed_ = true;
}

/// 1.5*(q1+q2) where q0>=q1>=q2>=0 are the eigenvalues of the momentum tensor sum{p_j[a]*p_j[b]}/sum{p_j**2}
/// normalized to 1. Return values are 1 for spherical, 3/4 for plane and 0 for linear events
double EventShapeVariables::sphericity() {
  if (!tensors_computed_)
    compTensorsAndVectors();
  return 1.5 * (eigenValues_[1] + eigenValues_[2]);
}

/// 1.5*q2 where q0>=q1>=q2>=0 are the eigenvalues of the momentum tensor sum{p_j[a]*p_j[b]}/sum{p_j**2}
/// normalized to 1. Return values are 0.5 for spherical and 0 for plane and linear events
double EventShapeVariables::aplanarity() {
  if (!tensors_computed_)
    compTensorsAndVectors();
  return 1.5 * eigenValues_[2];
}

/// 3.*(q0*q1+q0*q2+q1*q2) where q0>=q1>=q2>=0 are the eigenvalues of the momentum tensor sum{p_j[a]*p_j[b]}/sum{p_j**2}
/// normalized to 1. Return value is between 0 and 1
/// and measures the 3-jet structure of the event (C vanishes for a "perfect" 2-jet event)
double EventShapeVariables::C() {
  if (!tensors_computed_)
    compTensorsAndVectors();
  return 3. *
         (eigenValues_[0] * eigenValues_[1] + eigenValues_[0] * eigenValues_[2] + eigenValues_[1] * eigenValues_[2]);
}

/// 27.*(q0*q1*q2) where q0>=q1>=q2>=0 are the eigenvalues of the momemtum tensor sum{p_j[a]*p_j[b]}/sum{p_j**2}
/// normalized to 1. Return value is between 0 and 1
/// and measures the 4-jet structure of the event (D vanishes for a planar event)
double EventShapeVariables::D() {
  if (!tensors_computed_)
    compTensorsAndVectors();
  return 27. * eigenValues_[0] * eigenValues_[1] * eigenValues_[2];
}

//========================================================================================================

/// set number of Fox-Wolfram moments to compute
void EventShapeVariables::setFWmax(unsigned m) {
  fwmom_maxl_ = m;
  fwmom_computed_ = false;
  fwmom_ = std::vector<double>(fwmom_maxl_, 0.);
}

double EventShapeVariables::getFWmoment(unsigned l) {
  if (l > fwmom_maxl_)
    return 0.;

  if (!fwmom_computed_)
    computeFWmoments();

  return fwmom_[l];

}  // getFWmoment

const std::vector<double>& EventShapeVariables::getFWmoments() {
  if (!fwmom_computed_)
    computeFWmoments();

  return fwmom_;
}

void EventShapeVariables::computeFWmoments() {
  if (fwmom_computed_)
    return;

  double esum_total(0.);
  for (unsigned int i = 0; i < inputVectors_.size(); i++) {
    esum_total += inputVectors_[i].R();
  }  // i
  double esum_total_sq = esum_total * esum_total;

  for (unsigned int i = 0; i < inputVectors_.size(); i++) {
    double p_i = inputVectors_[i].R();
    if (p_i <= 0)
      continue;

    for (unsigned int j = 0; j <= i; j++) {
      double p_j = inputVectors_[j].R();
      if (p_j <= 0)
        continue;

      /// reduce computation by exploiting symmetry:
      /// all off-diagonal elements appear twice in the sum
      int symmetry_factor = 2;
      if (j == i)
        symmetry_factor = 1;
      double p_ij = p_i * p_j;
      double cosTheta = inputVectors_[i].Dot(inputVectors_[j]) / (p_ij);
      double pi_pj_over_etot2 = p_ij / esum_total_sq;

      /// compute higher legendre polynomials recursively
      /// need to keep track of two previous values
      double Pn1 = 0;
      double Pn2 = 0;
      for (unsigned n = 0; n < fwmom_maxl_; n++) {
        /// initial cases
        if (n == 0) {
          Pn2 = pi_pj_over_etot2;
          fwmom_[0] += Pn2 * symmetry_factor;
        } else if (n == 1) {
          Pn1 = pi_pj_over_etot2 * cosTheta;
          fwmom_[1] += Pn1 * symmetry_factor;
        } else {
          double Pn = ((2 * n - 1) * cosTheta * Pn1 - (n - 1) * Pn2) / n;
          fwmom_[n] += Pn * symmetry_factor;
          /// store new value
          Pn2 = Pn1;
          Pn1 = Pn;
        }
      }

    }  // j
  }    // i

  fwmom_computed_ = true;

}  // computeFWmoments

//========================================================================================================
