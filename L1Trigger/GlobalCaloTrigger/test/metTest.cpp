#include <iostream>
#include <cmath>

using std::cout;
using std::endl;

struct etmiss_vec {
	  unsigned mag;
	  unsigned phi;
};
etmiss_vec calculate_etmiss_vec (const int ex, const int ey) ;
unsigned phFine(unsigned ph) { 
  unsigned r = ph % 18;
  return (r<9 ? r : 17-r);
}

int main(int argc, char **argv)
{
  const unsigned maxbin=9;
  unsigned histTru[maxbin]={0,0,0,0,0,0,0,0,0};
  unsigned histGct[maxbin]={0,0,0,0,0,0,0,0,0};
  for (int ex=-50; ex<50; ex++) { for (int ey=-50; ey<50; ey++) {
    //if ((ex*ex+ey*ey)<100) {
    if ((ex*ex+ey*ey)<1000 && (ex*ex+ey*ey)>=500) {
      etmiss_vec met = calculate_etmiss_vec(ex,ey);
      unsigned phiF = phFine(met.phi);
      if (phiF<=maxbin) { histGct[phiF]++; } else { cout << " unexpected phi " << phiF << endl; }
      float phiVal = (atan2( (float) ey, (float) ex))/M_PI;
      if (phiVal<0) { phiVal += 2.0; }
      unsigned phiTrue = ((unsigned) (36.*phiVal));
      unsigned phiT = phFine(phiTrue);
      if (phiT<=maxbin) { histTru[phiT]++; } else { cout << " unexpected phi " << phiT << endl; }
      if (met.phi != phiTrue) {
        cout << "ex " << ex << " ey " << ey << " gct phi " << met.phi
                                            << " tru phi " << phiTrue << endl; 
      }
    }
  } }
  cout << "Phi bins Gct"; for (int b=0; b<maxbin; b++) { cout << "  " << histGct[b]; } cout << endl;
  cout << "Phi bins Tru"; for (int b=0; b<maxbin; b++) { cout << "  " << histTru[b]; } cout << endl;
  return 0;
}

etmiss_vec calculate_etmiss_vec (const int ex, const int ey) 
{
  //---------------------------------------------------------------------------------
  //
  // Calculates magnitude and direction of missing Et, given measured Ex and Ey.
  //
  // The algorithm used is suitable for implementation in hardware, using integer
  // multiplication, addition and comparison and bit shifting operations.
  //
  // Proceed in two stages. The first stage gives a result that lies between
  // 92% and 100% of the true Et, with the direction measured in 45 degree bins.
  // The final precision depends on the number of factors used in corrFact.
  // The present version with eleven factors gives a precision of 1% on Et, and
  // finds the direction to the nearest 5 degrees.
  //
  //---------------------------------------------------------------------------------
  etmiss_vec result;

  unsigned eneCoarse, phiCoarse;
  unsigned eneCorect, phiCorect;

  const unsigned root2fact = 181;
  const unsigned corrFact[11] = {24, 39, 51, 60, 69, 77, 83, 89, 95, 101, 106};
  const unsigned corrDphi[11] = { 0,  1,  1,  2,  2,  3,  3,  3,  3,   4,   4};

  bool s[3];
  unsigned Mx, My, Mw;

  unsigned Dx, Dy;
  unsigned Fx, Fy;
  unsigned eFact;

  unsigned b,phibin;
  bool midphi=false;

  // Here's the coarse calculation, with just one multiply operation
  //
  My = (ey<0 ? -ey : ey);
  Mx = (ex<0 ? -ex : ex);
  Mw = (((Mx+My)*root2fact)+0x80)>>8;

  s[0] = (ey<0);
  s[1] = (ex<0);
  s[2] = (My>Mx);

  phibin = 0; b = 0;
  for (int i=0; i<3; i++) {
    if (s[i]) { b=1-b;} phibin = 2*phibin + b;
  }

  unsigned m=(My>Mx ? My : Mx);
  eneCoarse = (Mw>m ? Mw : m );
  phiCoarse = phibin*9;

  // For the fine calculation we multiply both input components
  // by all the factors in the corrFact list in order to find
  // the required corrections to the energy and angle
  //
  for (eFact=0; eFact<10; eFact++) {
    Fx = Mx;
    Fy = My;
    Dx = (Mx*corrFact[eFact])>>8;
    Dy = (My*corrFact[eFact])>>8;
    if         ((Dx>=Fy) || (Dy>=Fx))         {midphi=false; break;}
    if ((Fx+Dx)>=(Fy-Dy) && (Fy+Dy)>=(Fx-Dx)) {midphi=true;  break;}
  }
  eneCorect = (eneCoarse*(128+eFact))>>7;
  if (midphi ^ (b==1)) {
    phiCorect = phiCoarse + 8 - corrDphi[eFact];
  } else {
    phiCorect = phiCoarse + corrDphi[eFact];
  }

  // Store the result of the calculation
  //
  result.mag = eneCorect;
  result.phi = phiCorect;

  return result;
}
