#ifndef rz_poly_h
#define rz_poly_h

/** \class magfieldparam::rz_poly
 *
 *
 *  \author V. Maroussov
 */

#include <vector>

namespace magfieldparam {

  //_______________________________________________________________________________
  struct poly_term {
    double coeff;
    int np[2];
  };

  //_______________________________________________________________________________
  typedef std::vector<poly_term> poly_vect;
  //_______________________________________________________________________________
  typedef std::vector<poly_vect> poly_arr;

  //_______________________________________________________________________________
  class rz_poly {  // represent a set of homogeneous polynomials

  private:
    poly_arr data;
    int max_nr, max_nz, n_active;
    double *r_pow;
    double *z_pow;
    bool *is_off;

  public:
    rz_poly() : data(), max_nr(0), max_nz(0), n_active(0), r_pow(nullptr), z_pow(nullptr), is_off(nullptr) {}

    rz_poly(int N);
    rz_poly(const rz_poly &S);
    ~rz_poly();

    void SetOFF(int npoly);
    void SetON(int npoly);
    void SetAllON(int npoly) {
      if (is_off)
        std::fill(is_off, is_off + data.size(), false);
    }

    rz_poly Diff(int nvar, bool keep_empty = false);
    rz_poly Int(int nvar);

    rz_poly &operator*=(double C);
    rz_poly &operator*=(double *C);

    double GetSVal(double r, double z, const double *C) const;
    double *GetVVal(double r, double z, double *rez_out = nullptr);

    int GetMaxRPow() const { return max_nr - 1; }
    int GetMaxZPow() const { return max_nz - 1; }
    int GetLength() const { return (int)data.size(); }
    int GetNActive() const { return n_active; }

    double *Expand(double *C);

    void Print();
  };
}  // namespace magfieldparam

#endif
