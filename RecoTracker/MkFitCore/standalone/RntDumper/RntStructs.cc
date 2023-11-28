#include "RntStructs.h"

#include <cstdio>

namespace {
  constexpr bool isFinite(float x) {
    const unsigned int mask = 0x7f800000;
    union {
      unsigned int l;
      float d;
    } v = {.d = x};
    return (v.l & mask) != mask;
  }

  // nan-guard, in place, return true if nan detected.
  bool ngr(float &f) {
    bool is_bad = !isFinite(f);
    if (is_bad)
      f = -999.0f;
    return is_bad;
  }
  bool ngr(RVec &v) {
    bool is_bad = ngr(v.fX);
    is_bad |= ngr(v.fY);
    is_bad |= ngr(v.fZ);
    return is_bad;
  }
  bool ngr(State &s) {
    bool is_bad = ngr(s.pos);
    is_bad |= ngr(s.mom);
    return is_bad;
  }
}  // namespace

bool BinSearch::nan_check() {
  has_nans = ngr(phi);
  has_nans |= ngr(dphi);
  has_nans |= ngr(q);
  has_nans |= ngr(dq);
  return has_nans;
}

void CandInfo::nan_check() {
  has_nans = ngr(ps_min);
  has_nans |= ngr(ps_max);
  has_nans |= bso.nan_check();
  has_nans |= bsn.nan_check();
}

void FailedPropInfo::nan_check() {
  has_nans = ngr(s_prev);
  has_nans |= ngr(s_final);
}
