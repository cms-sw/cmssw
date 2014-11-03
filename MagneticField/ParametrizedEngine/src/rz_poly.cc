#include <iostream>
#include "rz_poly.h"

using namespace std;
using namespace magfieldparam;

//_______________________________________________________________________________
rz_poly::rz_poly(int N)
{
   int nz, nr = 0, nv, nt;
   poly_term  v3;
   
   if (N < 2) N = 2;
   data.reserve(N);

   v3.coeff = 1.;
   v3.np[0] = 0;
   v3.np[1] = 1;
   
   data.push_back(poly_vect(1, v3));
   
   for (int m = 2; m <=N; ++m) {
      nz = m;
      nr = 0;
      nv = 0;
      v3.coeff = 1./m;
      v3.np[0] = nr;
      v3.np[1] = nz;

      nt = (m + 2) / 2;
      poly_vect v3x(nt, v3);
      
      while (nz >= 2) {
         nz -= 2;
         nr += 2;
         nv += 1;
         v3x[nv].coeff = -v3x[nv-1].coeff*(nz+1)*(nz+2)/(nr*nr);
         v3x[nv].np[0] = nr;
         v3x[nv].np[1] = nz;
      }
      data.push_back(v3x);
   }
   max_nr = nr+1;
   max_nz = N +1;
   r_pow = new double [max_nr];
   z_pow = new double [max_nz];
   fill(r_pow, r_pow+max_nr, 1.);
   fill(z_pow, z_pow+max_nz, 1.);
   
   n_active = data.size();
   is_off = new bool [n_active];
   fill(is_off, is_off+n_active, false);
}

//_______________________________________________________________________________
rz_poly::rz_poly(const rz_poly& S)
{
   data = S.data;
   max_nr = S.max_nr;
   max_nz = S.max_nz;
   n_active = S.n_active;

   if (max_nr) {
      r_pow = new double [max_nr];
      copy(S.r_pow, S.r_pow+max_nr, r_pow);
   } else r_pow = 0;

   if (max_nz) {
      z_pow = new double [max_nz];
      copy(S.z_pow, S.z_pow+max_nz, z_pow);
   } else z_pow = 0;

   if (S.is_off) {
      is_off = new bool [data.size()];
      copy(S.is_off, S.is_off+data.size(), is_off);
   } else is_off = 0;
}

//_______________________________________________________________________________
rz_poly::~rz_poly()
{
   if (is_off) delete [] is_off;
   if (r_pow)  delete [] r_pow;
   if (z_pow)  delete [] z_pow;
}

//_______________________________________________________________________________
void rz_poly::SetOFF(int npoly)
{
   if ((npoly < 0) || (npoly >= (int)data.size())) return;
   if (is_off[npoly]) return;
   is_off[npoly] = true;
   --n_active;
}

//_______________________________________________________________________________
void rz_poly::SetON(int npoly)
{
   if ((npoly < 0) || (npoly >= (int)data.size())) return;
   if (is_off[npoly]) {
      is_off[npoly] = false;
      ++n_active;
   }
}

//_______________________________________________________________________________
void rz_poly::Print()
{
   if (!data.size()) {
      cout << "The \"rz_poly\" object is NOT initialized!" << endl;
      return;
   }
   
   for (unsigned int ip = 0; ip < data.size(); ++ip) {
      cout << "Polynomial " << ip << " (size=" << data[ip].size() << ", status=";
      if (is_off[ip]) cout << "OFF):" << endl;
      else            cout << "ON):"  << endl;
      for (unsigned int it = 0; it < data[ip].size(); ++it) {
         cout << "\tnr="    << data[ip][it].np[0]
              << "\tnz="    << data[ip][it].np[1]
              << "\tcoeff=" << data[ip][it].coeff
              << endl;
      }
   }
}

//_______________________________________________________________________________
rz_poly rz_poly::Diff(int nvar, bool keep_empty)
{
//Return the derivative of original polynomial by variable nvar. 
//If keep_empty=true, resulting polynomial may contain zero basis terms,
//otherwise the resulting polynomial will be compressed, so that corresponding
//basis terms will be shifted relatively to the original polynomial.
//
   poly_term  v3;
   rz_poly p_out;
   p_out.data.reserve(data.size());
   
   bool *tmp_mask = new bool [data.size()];
   unsigned int  ind_tmp = 0;
   int   tmp_act = 0;
   
   for (unsigned int ip = 0; ip < data.size(); ++ip) {
      poly_vect v3x;
      v3x.reserve(data[ip].size());
      for (unsigned int it = 0; it < data[ip].size(); ++it) {
         v3 = data[ip][it];
         v3.coeff = data[ip][it].coeff * data[ip][it].np[nvar];
         if (v3.coeff != 0) {
            v3.np[nvar] = data[ip][it].np[nvar] - 1;
            v3x.push_back(v3);
         }
      }
      if (v3x.size() || keep_empty) {
         v3x.resize(v3x.size());
         p_out.data.push_back(v3x);
         tmp_mask[ind_tmp] = is_off[ip];
         ++ind_tmp;
         if (! is_off[ip]) ++tmp_act;
      }
   }
   
   p_out.data.resize(p_out.data.size());
   p_out.max_nr = max_nr;
   p_out.max_nz = max_nz;

   if (nvar == 0) --p_out.max_nr; else --p_out.max_nz;
   
   p_out.r_pow = new double [p_out.max_nr];
   copy(r_pow, r_pow+p_out.max_nr, p_out.r_pow);
   p_out.z_pow = new double [p_out.max_nz];
   copy(z_pow, z_pow+p_out.max_nz, p_out.z_pow);
   
   p_out.n_active = tmp_act;
   p_out.is_off = new bool [p_out.data.size()];
   copy(tmp_mask, tmp_mask+p_out.data.size(), p_out.is_off);
   
   delete [] tmp_mask;

   return p_out;
}

//_______________________________________________________________________________
rz_poly rz_poly::Int(int nvar)
{
//Return the integral of original polynomial by variable nvar
//
   rz_poly p_out;
   p_out.data = data;
   for (unsigned int ip = 0; ip < data.size(); ++ip) {
      for (unsigned int it = 0; it < data[ip].size(); ++it) {
         p_out.data[ip][it].coeff /= ++p_out.data[ip][it].np[nvar];
      }
   }
   
   p_out.max_nr = max_nr;
   p_out.max_nz = max_nz;

   if (nvar == 0) ++p_out.max_nr; else ++p_out.max_nz;
   
   p_out.r_pow = new double [p_out.max_nr];
   copy(r_pow, r_pow+max_nr, p_out.r_pow);
   p_out.z_pow = new double [p_out.max_nz];
   copy(z_pow, z_pow+max_nz, p_out.z_pow);
   
   if (nvar == 0) p_out.r_pow[max_nr] = p_out.r_pow[max_nr-1] * r_pow[1];
   else           p_out.z_pow[max_nz] = p_out.z_pow[max_nz-1] * z_pow[1];

   p_out.n_active = n_active;
   p_out.is_off = new bool [data.size()];
   copy(is_off, is_off+data.size(), p_out.is_off);
   
   return p_out;
}

//_______________________________________________________________________________
rz_poly& rz_poly::operator*=(double C)
{
//Multiply the polynomial by a constant. Skips terms that are switched off

   for (unsigned int ip = 0; ip < data.size(); ++ip) {
      if (is_off[ip]) continue;
      for (unsigned int it = 0; it < data[ip].size(); ++it) {
         data[ip][it].coeff *= C;
      }
   }
   return *this;
}

//_______________________________________________________________________________
rz_poly& rz_poly::operator*=(double *C)
{
//Multiply the polynomial by an array. Skips terms that are switched off

   for (unsigned int ip = 0; ip < data.size(); ++ip) {
      if (is_off[ip]) continue;
      for (unsigned int it = 0; it < data[ip].size(); ++it) {
         data[ip][it].coeff *= *C;
      }
      ++C;
   }
   return *this;
}

//_______________________________________________________________________________
double rz_poly::GetSVal(double r, double z, const double *C) const
{
//Return value of a polynomial, ignoring terms, that are switched off

   if (r_pow == 0) return 0.;
   
   double term, rez = 0.;
   int ip, it;
   
   if (r != r_pow[1]) for (ip = 1; ip < max_nr; ++ip) r_pow[ip] = r*r_pow[ip-1];
   if (z != z_pow[1]) for (ip = 1; ip < max_nz; ++ip) z_pow[ip] = z*z_pow[ip-1];

   for (ip = 0; ip < (int)data.size(); ++ip) {
      if (is_off[ip]) continue;
      term = 0.;
      for (it = 0; it < (int)data[ip].size(); ++it) {
         term += data[ip][it].coeff
               * r_pow[data[ip][it].np[0]]
               * z_pow[data[ip][it].np[1]];
      }
      rez += *C * term;
      ++C;
   }
   
   return rez;
}

//_______________________________________________________________________________
double *rz_poly::GetVVal(double r, double z, double *rez_out)
{
//return an array of doubleype with values of the basis functions in the point
//(r,z). In a case if rez_out != 0, the rez_out array must be long enough to fit
//in n_active elements. In a case if rez_out == 0, a new array of n_active length
//is created; it is in user's responsibility to free the memory after all;
//
   if (r_pow == 0) return 0;
   
   double term;
   int ip, it;

   double *rez;
   if (rez_out) rez = rez_out;
   else rez = new double [n_active];

   double *pnt = rez;
   
   if (r != r_pow[1]) for (ip = 1; ip < max_nr; ++ip) r_pow[ip] = r*r_pow[ip-1];
   if (z != z_pow[1]) for (ip = 1; ip < max_nz; ++ip) z_pow[ip] = z*z_pow[ip-1];

   for (ip = 0; ip < (int)data.size(); ++ip) {
      if (is_off[ip]) continue;
      term = 0.;
      for (it = 0; it < (int)data[ip].size(); ++it) {
         term += data[ip][it].coeff
               * r_pow[data[ip][it].np[0]]
               * z_pow[data[ip][it].np[1]];
      }
      *pnt = term;
      ++pnt;
   }
   
   return rez;
}

//_______________________________________________________________________________
double *rz_poly::Expand(double *C)
{
//Take C[n_active] - reduced vector of coefficients and return pointer to 
//expanded vector of coefficients of the data.size() length. It is in user's
//responsibility to free the memory after all;
//
   double *rez = new double [data.size()];
   for (int ip = 0; ip < (int)data.size(); ++ip) {
      if (is_off[ip]) rez[ip] = 0.;
      else {
         rez[ip] = *C;
         ++C;
      }
   }
   return rez;
}

