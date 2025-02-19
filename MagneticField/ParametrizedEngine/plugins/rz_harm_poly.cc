#include <typeinfo>
#include "rz_harm_poly.h"
#include <cstdlib>

using namespace magfieldparam;

/////////////////////////////////////////////////////////////////////////////////
//                                                                             //
//  Harmonic homogeneous polynomials in cylindrical system.                    //
//                                                                             //
/////////////////////////////////////////////////////////////////////////////////

//_______________________________________________________________________________
unsigned rz_harm_poly::Cnt    = 0;      //Number of the "rz_harm_poly" objects
double   rz_harm_poly::phival = -11111.;//Last phi value used
bool     rz_harm_poly::phi_set = false; //TRUE if phi value is set
unsigned rz_harm_poly::MaxM   = 0;      //Max. M among "rz_harm_poly" objects

unsigned   rz_harm_poly::TASize  = 0;   //TrigArr size
trig_pair *rz_harm_poly::TrigArr = 0;   //Array with angular data

//_______________________________________________________________________________
rz_harm_poly::rz_harm_poly(const unsigned N)
{
//Constructor for rz_harm_poly of length N. The polynomial P(r,z) is normalized
//in such a way that dP/dz(r=0,z)=z^(N-1)
//
   unsigned nz = N, nr = 0, nv = 0;
   poly2d_term v3(1./N, nr, nz);

   data = std::vector<poly2d_term>((N + 2) / 2, v3);

   while (nz >= 2) {
      nz -= 2;
      nr += 2;
      nv += 1;
      data[nv].coeff = -data[nv-1].coeff*(nz+1)*(nz+2)/(nr*nr);
      data[nv].np[0] = nr;
      data[nv].np[1] = nz;
   }
   max_pwr = N;
   if (max_pwr > NPwr) {
      NPwr = max_pwr;
      rz_set = false;
      phi_set = false;
   }
   L = N;
   M = 0;
   poly2d_base_set.insert(this);
   ++Cnt;
}

//_______________________________________________________________________________
rz_harm_poly::~rz_harm_poly()
{
   if (--Cnt) {
     if (std::abs(M) >= int(MaxM)) { //a number of objects still left
         M = 0;
         MaxM = GetMaxM();
      }
   } else { //last instance -> memory cleanup
      if (TrigArr) delete [] TrigArr;
      TrigArr = 0;
      TASize = 0;
      MaxM = 0;
      phival = -11111.;
      phi_set = false;
   }
}

//_______________________________________________________________________________
int rz_harm_poly::GetMaxM()
{
//Return max abs(M) for all rz_harm_poly objects created
//
   int M_cur, M_max = 0;
   std::set<poly2d_base*>::iterator it;
   for (it = poly2d_base_set.begin(); it != poly2d_base_set.end(); ++it) {
      if (typeid(**it) == typeid(rz_harm_poly)) {
         M_cur = std::abs(((rz_harm_poly*)(*it))->M);
         if (M_cur > M_max) M_max = M_cur;
      }
   }
   return M_max;
}

//_______________________________________________________________________________
void rz_harm_poly::SetPhi(const double phi)
{
//Set value of the angle argument, adjust the TrigArr size if neccessary
//and fill TrigArr if the phi value is changed
//
   if (MaxM >= TASize) { SetTrigArrSize(MaxM+1); FillTrigArr(phi);}
   else if (phi != phival) FillTrigArr(phi);
   phival = phi;
   phi_set = true;
}

//_______________________________________________________________________________
void rz_harm_poly::SetTrigArrSize(const unsigned N)
{
//Increase TrigArr size if neccessary
//
   if (N <= TASize) return;
   if (TrigArr) delete [] TrigArr;
   TrigArr = new trig_pair [N];
   (*TrigArr) = trig_pair(1., 0.);
   TASize = N;
   phi_set = false;
}

//_______________________________________________________________________________
void rz_harm_poly::FillTrigArr(const double phi)
{
//Fill TrigArr with trig_pair(jp*phi)
   if (!TrigArr) return;
   trig_pair tp(phi);
   TrigArr[1] = tp;
   for (unsigned jp = 2; jp <= MaxM; ++jp) TrigArr[jp] = TrigArr[jp-1].Add(tp);
}

//_______________________________________________________________________________
void rz_harm_poly::PrintTrigArr(std::ostream &out, const std::streamsize prec)
{
   out << "TrigArr: TASize = " << TASize
       << "\tMaxM = " << MaxM << std::endl;
   if (TrigArr) {
      if (MaxM < TASize) {
         unsigned jm;
         std::streamsize old_prec = out.precision(), wdt = prec+7;
         out.precision(prec);
         out << "M:     ";
         for (jm = 0; jm <= MaxM; ++jm) {
            out << std::setw(wdt) << std::left << jm;
         }
         out << "|\nCos_M: ";
         for (jm = 0; jm <= MaxM; ++jm) {
            out << std::setw(wdt) << std::left << TrigArr[jm].CosPhi;
         }
         out << "|\nSin_M: ";
         for (jm = 0; jm <= MaxM; ++jm) {
            out << std::setw(wdt) << std::left << TrigArr[jm].SinPhi;
         }
         out << "|" << std::endl;
         out.precision(old_prec);
      } else {
         out << "\tTrigArr size is not adjusted." << std::endl;
      }
   } else {
      out << "\tTrigArr is not allocated." << std::endl;
   }
}

//_______________________________________________________________________________
rz_harm_poly rz_harm_poly::LadderUp()
{
//Return a polynomial with increased M
//
   rz_harm_poly p_out; p_out.data.reserve(2*L);
   unsigned it;
   poly2d_term term;

   //In 2 passes (for-cycles) to get terms in z-descending order
   for(it = 0; it < data.size(); ++it) {
      term = data[it];
      if (term.np[0]) {
         term.coeff *= int(term.np[0]) - M;
         --term.np[0];
         ++term.np[1];
         p_out.data.push_back(term);
      }
   }
   for(it = 0; it < data.size(); ++it) {
      term = data[it];
      if (term.np[1]) {
         term.coeff *= -(int)term.np[1];
         --term.np[1];
         ++term.np[0];
         p_out.data.push_back(term);
      }
   }
   p_out.Collect();
   if (p_out.data.size()) {
      p_out.L = L;
      p_out.M = M+1;
      if (std::abs(p_out.M) > int(MaxM)) MaxM = std::abs(p_out.M);
      p_out.Scale(1./sqrt(double((L-M)*(L+M+1))));
   }
   return p_out;
}

//_______________________________________________________________________________
rz_harm_poly rz_harm_poly::LadderDwn()
{
//Return a polynomial with decreased M
//
   rz_harm_poly p_out; p_out.data.reserve(2*L);
   unsigned it;
   poly2d_term term;

   //In 2 passes (for-cycles) to get terms in z-descending order
   for(it = 0; it < data.size(); ++it) {
      term = data[it];
      if (term.np[0]) {
         term.coeff *= -int(term.np[0]) - M;
         --term.np[0];
         ++term.np[1];
         p_out.data.push_back(term);
      }
   }
   for(it = 0; it < data.size(); ++it) {
      term = data[it];
      if (term.np[1]) {
         term.coeff *= term.np[1];
         --term.np[1];
         ++term.np[0];
         p_out.data.push_back(term);
      }
   }
   p_out.Collect();
   if (p_out.data.size()) {
      p_out.L = L;
      p_out.M = M-1;
      if (std::abs(p_out.M) > int(MaxM)) MaxM = std::abs(p_out.M);
      p_out.Scale(1./sqrt(double((L+M)*(L-M+1))));
   }
   return p_out;
}
   
