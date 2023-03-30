#include "poly2d_base.h"

using namespace magfieldparam;

/////////////////////////////////////////////////////////////////////////////////
//                                                                             //
//  The "poly2d_term" represent a term of a polynomial of 2 variables.         //
//                                                                             //
/////////////////////////////////////////////////////////////////////////////////

//_______________________________________________________________________________
void poly2d_term::Print(std::ostream &out, bool first_term) {
  if (first_term)
    out << coeff;
  else if (coeff > 0.)
    out << " + " << coeff;
  else
    out << " - " << -coeff;
  if (np[0] != 0) {
    out << "*r";
    if (np[0] != 1)
      out << "^" << np[0];
  }
  if (np[1] != 0) {
    out << "*z";
    if (np[1] != 1)
      out << "^" << np[1];
  }
}

/////////////////////////////////////////////////////////////////////////////////
//                                                                             //
//  Base class that represent a polynomial of 2 variables. It isn't supposed   //
//  to be used directly and provides no way of setting coefficients directly.  //
//  Such methods must be defined in derived classes.                           //
//                                                                             //
/////////////////////////////////////////////////////////////////////////////////

//_______________________________________________________________________________
double poly2d_base::rval = 0.;  //Last values of r and z used
double poly2d_base::zval = 0.;

double **poly2d_base::rz_pow = nullptr;  //table with calculated r^n*z^m values
unsigned poly2d_base::NTab = 0;          //rz_pow table size
unsigned poly2d_base::NPwr = 0;          //max power in use by CLASS

bool poly2d_base::rz_set = false;

const double poly2d_base::MIN_COEFF = DBL_EPSILON;  //Thresh. for coeff == 0
std::set<poly2d_base *> poly2d_base::poly2d_base_set;

//_______________________________________________________________________________
poly2d_base::~poly2d_base() {
  poly2d_base_set.erase(poly2d_base_set.find(this));
  if (!poly2d_base_set.empty()) {  //some objects left
    if (max_pwr >= NPwr)
      NPwr = GetMaxPow();
  } else {
    if (rz_pow) {
      delete[] rz_pow[0];  //deleting the last instance -> memory cleanup
      delete[] rz_pow;
    }
    rz_pow = nullptr;
    rval = zval = 0.;
    NPwr = 0;
    NTab = 0;
    rz_set = false;
    //      poly2d_base_set.resize(0);
  }
}

//_______________________________________________________________________________
void poly2d_base::SetTabSize(const unsigned N) {
  if (N <= NTab)
    return;
  if (rz_pow) {
    delete[] rz_pow[0];
    delete[] rz_pow;
  }
  rz_pow = new double *[N];
  unsigned jr, dN = N * (N + 1) / 2;
  rz_pow[0] = new double[dN];
  memset(rz_pow[0], 0, dN * sizeof(double));
  rz_pow[0][0] = 1.;
  for (jr = 1, dN = N; jr < N; ++jr, --dN) {
    rz_pow[jr] = rz_pow[jr - 1] + dN;
  }
  rval = zval = 0.;
  NTab = N;
}

//_______________________________________________________________________________
void poly2d_base::FillTable(const double r, const double z) {
  if (!rz_pow)
    return;
  unsigned jr, jz;
  for (jz = 1; jz <= NPwr; ++jz)
    rz_pow[0][jz] = z * rz_pow[0][jz - 1];
  for (jr = 1; jr <= NPwr; ++jr) {
    for (jz = 0; jz <= (NPwr - jr); ++jz) {
      rz_pow[jr][jz] = r * rz_pow[jr - 1][jz];
    }
  }
}

//_______________________________________________________________________________
int poly2d_base::GetMaxPow() {
  int curp, maxp = 0;
  std::set<poly2d_base *>::iterator it;

  for (it = poly2d_base_set.begin(); it != poly2d_base_set.end(); ++it) {
    curp = (*it)->max_pwr;
    if (curp > maxp)
      maxp = curp;
  }
  return maxp;
}

//_______________________________________________________________________________
void poly2d_base::AdjustTab() {
  NPwr = GetMaxPow();
  if (NPwr >= NTab)
    SetTabSize(NPwr + 1);
}

//_______________________________________________________________________________
void poly2d_base::PrintTab(std::ostream &out, const std::streamsize prec) {
  out << "poly2d_base table size NTab = " << NTab << "\tmax. power NPwr = " << NPwr << std::endl;
  if (rz_pow) {
    if (NPwr < NTab) {
      std::streamsize old_prec = out.precision(), wdt = prec + 7;
      out.precision(prec);
      out << "Table content:" << std::endl;
      unsigned jr, jz;
      for (jr = 0; jr <= NPwr; ++jr) {
        for (jz = 0; jz <= (NPwr - jr); ++jz) {
          out << std::setw(wdt) << std::left << rz_pow[jr][jz];
        }
        out << "|" << std::endl;
      }
      out.precision(old_prec);
    } else {
      out << "\tTable size is not adjusted." << std::endl;
    }
  } else {
    out << "\tTable is not allocated." << std::endl;
  }
}

//_______________________________________________________________________________
void poly2d_base::SetPoint(const double r, const double z) {
  if (!Count())
    return;
  if (NPwr >= NTab) {
    SetTabSize(NPwr + 1);
    FillTable(r, z);
  } else if ((r != rval) || (z != zval))
    FillTable(r, z);
  rz_set = true;
}

//_______________________________________________________________________________
double poly2d_base::Eval() {
  double S = 0.;
  for (unsigned j = 0; j < data.size(); ++j)
    S += data[j].coeff * rz_pow[data[j].np[0]][data[j].np[1]];
  return S;
}

//_______________________________________________________________________________
void poly2d_base::Collect() {
  if (data.empty())
    return;

  unsigned j1, j2, rpow, zpow, noff = 0, jend = data.size();
  double C;
  std::vector<bool> mask(jend, false);
  max_pwr = 0;

  for (j1 = 0; j1 < jend; ++j1) {
    if (mask[j1])
      continue;
    C = data[j1].coeff;
    rpow = data[j1].np[0];
    zpow = data[j1].np[1];
    for (j2 = j1 + 1; j2 < jend; ++j2) {
      if (mask[j2])
        continue;
      if ((rpow == data[j2].np[0]) && (zpow == data[j2].np[1])) {
        C += data[j2].coeff;
        mask[j2] = true;
        ++noff;
      }
    }
    if (fabs(C) > MIN_COEFF) {
      data[j1].coeff = C;
      if ((rpow = rpow + zpow) > max_pwr)
        max_pwr = rpow;
    } else {
      mask[j1] = true;
      ++noff;
    }
  }
  std::vector<poly2d_term> newdata;
  newdata.reserve(jend - noff);
  for (j1 = 0; j1 < jend; ++j1) {
    if (!(mask[j1]))
      newdata.push_back(data[j1]);
  }
  data.swap(newdata);
}

//_______________________________________________________________________________
void poly2d_base::Print(std::ostream &out, const std::streamsize prec) {
  if (data.empty()) {
    out << "\"poly2d_base\" object contains no terms." << std::endl;
    return;
  }
  out << data.size() << " terms; max. degree = " << max_pwr << ":" << std::endl;
  std::streamsize old_prec = out.precision();
  out.precision(prec);
  data[0].Print(out);
  for (unsigned it = 1; it < data.size(); ++it) {
    data[it].Print(out, false);
  }
  out << std::endl;
  out.precision(old_prec);
}

//_______________________________________________________________________________
void poly2d_base::Diff(int nvar) {
  //differentiate the polynomial by variable nvar.
  //
  poly2d_term v3;
  std::vector<poly2d_term> newdata;
  newdata.reserve(data.size());
  unsigned cur_pwr = 0, maxp = 0, oldp = max_pwr;
  for (unsigned it = 0; it < data.size(); ++it) {
    v3 = data[it];
    v3.coeff *= v3.np[nvar];
    if (v3.coeff != 0.) {
      --v3.np[nvar];
      newdata.push_back(v3);
      if ((cur_pwr = v3.np[0] + v3.np[1]) > maxp)
        maxp = cur_pwr;
    }
  }
  newdata.resize(newdata.size());
  max_pwr = maxp;
  data.swap(newdata);
  if (oldp >= NPwr)
    NPwr = GetMaxPow();
}

//_______________________________________________________________________________
void poly2d_base::Int(int nvar) {
  //Integrate the polynomial by variable# nvar. Doesn't remove terms
  //with zero coefficients; if you suspect they can appear, use Compress()
  //after the integration.
  //
  for (unsigned it = 0; it < data.size(); ++it) {
    data[it].coeff /= ++data[it].np[nvar];
  }
  ++max_pwr;
  if (max_pwr > NPwr)
    NPwr = GetMaxPow();
}

//_______________________________________________________________________________
void poly2d_base::IncPow(int nvar) {
  //Multiply the polynomial by variable# nvar
  //
  for (unsigned it = 0; it < data.size(); ++it) {
    ++data[it].np[nvar];
  }
  ++max_pwr;
  if (max_pwr > NPwr)
    NPwr = GetMaxPow();
}

//_______________________________________________________________________________
void poly2d_base::DecPow(int nvar) {
  //Divide the polynomial by variable# nvar. Remove terms with zero coefficients
  //and also terms where the initial power of nvar is equal zero
  //
  poly2d_term v3;
  std::vector<poly2d_term> newdata;
  newdata.reserve(data.size());
  unsigned cur_pwr = 0, maxp = 0, oldp = max_pwr;
  for (unsigned it = 0; it < data.size(); ++it) {
    v3 = data[it];
    if ((v3.coeff != 0.) && (v3.np[nvar] > 0)) {
      --v3.np[nvar];
      newdata.push_back(v3);
      if ((cur_pwr = v3.np[0] + v3.np[1]) > maxp)
        maxp = cur_pwr;
    }
  }
  newdata.resize(newdata.size());
  max_pwr = maxp;
  data.swap(newdata);
  if (oldp >= NPwr)
    NPwr = GetMaxPow();
}

//_______________________________________________________________________________
void poly2d_base::Scale(const double C) {
  //Multiply the polynomial by a constant.
  //
  if (C != 0.) {
    for (unsigned it = 0; it < data.size(); ++it) {
      data[it].coeff *= C;
    }
  } else
    data.resize(0);
}
