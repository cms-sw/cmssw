#include <cassert>

#include "CalibCalorimetry/HcalAlgos/interface/ConstantStepOdeSolver.h"

inline static double interpolateLinear(const double x, const double f0, const double f1) {
  return f0 * (1.0 - x) + f1 * x;
}

double ConstantStepOdeSolver::getPeakTime(const unsigned which) const {
  if (which >= dim_)
    throw cms::Exception("In ConstantStepOdeSolver::getPeakTime: index out of range");
  if (runLen_ < 3)
    throw cms::Exception("In ConstantStepOdeSolver::getPeakTime: not enough data");

  const double* hbuf = &historyBuffer_[which];
  double maxval = hbuf[0];
  unsigned maxind = 0;
  for (unsigned i = 1; i < runLen_; ++i)
    if (hbuf[dim_ * i] > maxval) {
      maxval = hbuf[dim_ * i];
      maxind = i;
    }
  if (maxind == 0U)
    return 0.0;
  if (maxind == runLen_ - 1U)
    return dt_ * maxind;
  const double l = hbuf[dim_ * (maxind - 1U)];
  const double r = hbuf[dim_ * (maxind + 1U)];
  if (l < maxval || r < maxval)
    return dt_ * (maxind + (l - r) / 2.0 / (l + r - 2.0 * maxval));
  else
    return dt_ * maxind;
}

double ConstantStepOdeSolver::getIntegrated(const unsigned which, const unsigned idx) const {
  if (which >= dim_ || idx >= runLen_)
    throw cms::Exception("In ConstantStepOdeSolver::getIntegrated: index out of range");
  if (lastIntegrated_ != which)
    (const_cast<ConstantStepOdeSolver*>(this))->integrateCoordinate(which);
  return chargeBuffer_[idx];
}

void ConstantStepOdeSolver::integrateCoordinate(const unsigned which) {
  if (runLen_ < 4)
    throw cms::Exception("In ConstantStepOdeSolver::integrateCoordinate: not enough data");
  if (chargeBuffer_.size() < runLen_)
    chargeBuffer_.resize(runLen_);
  double* integ = &chargeBuffer_[0];
  const double* coord = &historyBuffer_[which];

  integ[0] = 0.0;
  integ[1] = coord[dim_ * 0] * (3.0 / 8.0) + coord[dim_ * 1] * (19.0 / 24.0) + coord[dim_ * 2] * (-5.0 / 24.0) +
             coord[dim_ * 3] * (1.0 / 24.0);
  long double sum = integ[1];
  const unsigned rlenm1 = runLen_ - 1U;
  for (unsigned i = 2; i < rlenm1; ++i) {
    sum += (coord[dim_ * (i - 2U)] * (-1.0 / 24.0) + coord[dim_ * (i - 1U)] * (13.0 / 24.0) +
            coord[dim_ * i] * (13.0 / 24.0) + coord[dim_ * (i + 1U)] * (-1.0 / 24.0));
    integ[i] = sum;
  }
  sum += (coord[dim_ * rlenm1] * (3.0 / 8.0) + coord[dim_ * (rlenm1 - 1U)] * (19.0 / 24.0) +
          coord[dim_ * (rlenm1 - 2U)] * (-5.0 / 24.0) + coord[dim_ * (rlenm1 - 3U)] * (1.0 / 24.0));
  integ[rlenm1] = sum;
  if (dt_ != 1.0)
    for (unsigned i = 1; i < runLen_; ++i)
      integ[i] *= dt_;
  lastIntegrated_ = which;
}

void ConstantStepOdeSolver::writeHistory(std::ostream& os, const double dt, const bool cubic) const {
  os << "# " << this->methodName() << '\n';
  if (dim_ && runLen_) {
    if (dt == dt_) {
      for (unsigned ipt = 0; ipt < runLen_; ++ipt) {
        os << getTime(ipt);
        for (unsigned which = 0; which < dim_; ++which)
          os << ' ' << getCoordinate(which, ipt);
        os << '\n';
      }
    } else {
      const double tmax = lastMaxT();
      for (unsigned i = 0;; ++i) {
        const double t = i * dt;
        if (t > tmax)
          break;
        os << t;
        for (unsigned which = 0; which < dim_; ++which)
          os << ' ' << interpolateCoordinate(which, t, cubic);
        os << '\n';
      }
    }
  }
}

void ConstantStepOdeSolver::writeIntegrated(std::ostream& os,
                                            const unsigned which,
                                            const double dt,
                                            const bool cubic) const {
  os << "# " << this->methodName() << '\n';
  if (dim_ && runLen_) {
    if (dt == dt_) {
      for (unsigned ipt = 0; ipt < runLen_; ++ipt)
        os << getTime(ipt) << ' ' << getIntegrated(which, ipt) << '\n';
    } else {
      const double tmax = lastMaxT();
      for (unsigned i = 0;; ++i) {
        const double t = i * dt;
        if (t > tmax)
          break;
        os << t << ' ' << interpolateIntegrated(which, t, cubic) << '\n';
      }
    }
  }
}

ConstantStepOdeSolver::ConstantStepOdeSolver(const ConstantStepOdeSolver& r)
    : rhs_(nullptr),
      dt_(r.dt_),
      dim_(r.dim_),
      runLen_(r.runLen_),
      lastIntegrated_(r.lastIntegrated_),
      historyBuffer_(r.historyBuffer_),
      chargeBuffer_(r.chargeBuffer_) {
  if (r.rhs_)
    rhs_ = r.rhs_->clone();
}

ConstantStepOdeSolver& ConstantStepOdeSolver::operator=(const ConstantStepOdeSolver& r) {
  if (this != &r) {
    delete rhs_;
    rhs_ = nullptr;
    dt_ = r.dt_;
    dim_ = r.dim_;
    runLen_ = r.runLen_;
    lastIntegrated_ = r.lastIntegrated_;
    historyBuffer_ = r.historyBuffer_;
    chargeBuffer_ = r.chargeBuffer_;
    if (r.rhs_)
      rhs_ = r.rhs_->clone();
  }
  return *this;
}

void ConstantStepOdeSolver::truncateCoordinate(const unsigned which, const double minValue, const double maxValue) {
  if (which >= dim_)
    throw cms::Exception("In ConstantStepOdeSolver::truncateCoordinate: index out of range");
  if (minValue > maxValue)
    throw cms::Exception("In ConstantStepOdeSolver::truncateCoordinate: invalid truncation range");

  double* buf = &historyBuffer_[which];
  for (unsigned i = 0; i < runLen_; ++i) {
    if (buf[dim_ * i] < minValue)
      buf[dim_ * i] = minValue;
    else if (buf[dim_ * i] > maxValue)
      buf[dim_ * i] = maxValue;
  }
}

double ConstantStepOdeSolver::interpolateCoordinate(const unsigned which, const double t, const bool cubic) const {
  if (which >= dim_)
    throw cms::Exception("In ConstantStepOdeSolver::interpolateCoordinate: index out of range");
  if (runLen_ < 2U || (cubic && runLen_ < 4U))
    throw cms::Exception("In ConstantStepOdeSolver::interpolateCoordinate: not enough data");
  const double maxt = runLen_ ? dt_ * (runLen_ - 1U) : 0.0;
  if (t < 0.0 || t > maxt)
    throw cms::Exception("In ConstantStepOdeSolver::interpolateCoordinate: time out of range");

  const double* arr = &historyBuffer_[0];
  if (t == 0.0)
    return arr[which];
  else if (t == maxt)
    return arr[which + dim_ * (runLen_ - 1U)];

  // Translate time into timestep units
  const double tSteps = t / dt_;
  unsigned nLow = tSteps;
  if (nLow >= runLen_ - 1)
    nLow = runLen_ - 2;
  double x = tSteps - nLow;

  if (cubic) {
    unsigned i0 = 0;
    if (nLow == runLen_ - 2) {
      i0 = nLow - 2U;
      x += 2.0;
    } else if (nLow) {
      i0 = nLow - 1U;
      x += 1.0;
    }
    const double* base = arr + (which + dim_ * i0);
    return interpolateLinear(x * (3.0 - x) / 2.0,
                             interpolateLinear(x / 3.0, base[0], base[dim_ * 3]),
                             interpolateLinear(x - 1.0, base[dim_], base[dim_ * 2]));
  } else
    return interpolateLinear(x, arr[which + dim_ * nLow], arr[which + dim_ * (nLow + 1U)]);
}

double ConstantStepOdeSolver::interpolateIntegrated(const unsigned which, const double t, const bool cubic) const {
  if (which >= dim_)
    throw cms::Exception("In ConstantStepOdeSolver::interpolateIntegrated: index out of range");
  if (runLen_ < 2U || (cubic && runLen_ < 4U))
    throw cms::Exception("In ConstantStepOdeSolver::interpolateIntegrated: not enough data");
  const double maxt = runLen_ ? dt_ * (runLen_ - 1U) : 0.0;
  if (t < 0.0 || t > maxt)
    throw cms::Exception("In ConstantStepOdeSolver::interpolateIntegrated: time out of range");
  if (lastIntegrated_ != which)
    (const_cast<ConstantStepOdeSolver*>(this))->integrateCoordinate(which);

  const double* buf = &chargeBuffer_[0];
  if (t == 0.0)
    return buf[0];
  else if (t == maxt)
    return buf[runLen_ - 1U];

  // Translate time into timestep units
  const double tSteps = t / dt_;
  unsigned nLow = tSteps;
  if (nLow >= runLen_ - 1)
    nLow = runLen_ - 2;
  double x = tSteps - nLow;

  if (cubic) {
    unsigned i0 = 0;
    if (nLow == runLen_ - 2) {
      i0 = nLow - 2U;
      x += 2.0;
    } else if (nLow) {
      i0 = nLow - 1U;
      x += 1.0;
    }
    const double* base = buf + i0;
    return interpolateLinear(x * (3.0 - x) / 2.0,
                             interpolateLinear(x / 3.0, base[0], base[3]),
                             interpolateLinear(x - 1.0, base[1], base[2]));
  } else
    return interpolateLinear(x, buf[nLow], buf[nLow + 1U]);
}

void ConstantStepOdeSolver::setHistory(const double dt, const double* data, const unsigned dim, const unsigned runLen) {
  const unsigned len = dim * runLen;
  if (!len)
    return;
  if (dt <= 0.0)
    throw cms::Exception("In ConstantStepOdeSolver::setHistory: can not run backwards in time");
  assert(data);
  const unsigned sz = dim * (runLen + 1U);
  if (historyBuffer_.size() < sz)
    historyBuffer_.resize(sz);
  dt_ = dt;
  dim_ = dim;
  runLen_ = runLen;
  lastIntegrated_ = dim_;
  double* arr = &historyBuffer_[0];
  for (unsigned i = 0; i < len; ++i)
    *arr++ = *data++;
  for (unsigned i = 0; i < dim; ++i)
    *arr++ = 0.0;
}

void ConstantStepOdeSolver::run(const double* initialConditions,
                                const unsigned lenInitialConditions,
                                const double dt,
                                const unsigned nSteps) {
  if (!nSteps || !lenInitialConditions)
    return;
  if (!rhs_)
    throw cms::Exception("In ConstantStepOdeSolver::run: ODE right hand side has not been set");
  if (dt <= 0.0)
    throw cms::Exception("In ConstantStepOdeSolver::run: can not run backwards in time");
  assert(initialConditions);
  dt_ = dt;
  dim_ = lenInitialConditions;
  lastIntegrated_ = dim_;
  runLen_ = nSteps + 1U;
  const unsigned sz = (runLen_ + 1U) * dim_;
  if (historyBuffer_.size() < sz)
    historyBuffer_.resize(sz);
  double* arr = &historyBuffer_[0];
  for (unsigned i = 0; i < lenInitialConditions; ++i)
    arr[i] = initialConditions[i];
  double* stepBuffer = arr + runLen_ * dim_;

  for (unsigned i = 0; i < nSteps; ++i, arr += lenInitialConditions) {
    const double t = i * dt;
    this->step(t, dt, arr, lenInitialConditions, stepBuffer);
    double* next = arr + lenInitialConditions;
    for (unsigned i = 0; i < lenInitialConditions; ++i)
      next[i] = arr[i] + stepBuffer[i];
  }
}

void EulerOdeSolver::step(
    const double t, const double dt, const double* x, const unsigned lenX, double* coordIncrement) const {
  rhs_->calc(t, x, lenX, coordIncrement);
  for (unsigned i = 0; i < lenX; ++i)
    coordIncrement[i] *= dt;
}

void RK2::step(const double t, const double dt, const double* x, const unsigned lenX, double* coordIncrement) const {
  const double halfstep = dt / 2.0;
  if (buf_.size() < lenX)
    buf_.resize(lenX);
  double* midpoint = &buf_[0];
  rhs_->calc(t, x, lenX, midpoint);
  for (unsigned i = 0; i < lenX; ++i) {
    midpoint[i] *= halfstep;
    midpoint[i] += x[i];
  }
  rhs_->calc(t + halfstep, midpoint, lenX, coordIncrement);
  for (unsigned i = 0; i < lenX; ++i)
    coordIncrement[i] *= dt;
}

void RK4::step(const double t, const double dt, const double* x, const unsigned lenX, double* coordIncrement) const {
  const double halfstep = dt / 2.0;
  if (buf_.size() < 4 * lenX)
    buf_.resize(4 * lenX);
  double* k1x = &buf_[0];
  double* k2x = k1x + lenX;
  double* k3x = k2x + lenX;
  double* k4x = k3x + lenX;
  rhs_->calc(t, x, lenX, k1x);
  for (unsigned i = 0; i < lenX; ++i)
    coordIncrement[i] = x[i] + halfstep * k1x[i];
  rhs_->calc(t + halfstep, coordIncrement, lenX, k2x);
  for (unsigned i = 0; i < lenX; ++i)
    coordIncrement[i] = x[i] + halfstep * k2x[i];
  rhs_->calc(t + halfstep, coordIncrement, lenX, k3x);
  for (unsigned i = 0; i < lenX; ++i)
    coordIncrement[i] = x[i] + dt * k3x[i];
  rhs_->calc(t + dt, coordIncrement, lenX, k4x);
  for (unsigned i = 0; i < lenX; ++i)
    coordIncrement[i] = dt / 6.0 * (k1x[i] + 2.0 * (k2x[i] + k3x[i]) + k4x[i]);
}
