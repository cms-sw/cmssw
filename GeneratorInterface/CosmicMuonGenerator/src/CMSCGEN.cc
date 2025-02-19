//
// CMSCGEN.cc   version 3.0    Thomas Hebbeker 2007-05-15  
//
// implemented in CMSSW by P. Biallass 2007-05-28 
// see header for documentation and CMS internal note 2007 "Improved Parametrization of the Cosmic Muon Flux for the generator CMSCGEN" by Biallass + Hebbeker
//

#include <CLHEP/Random/RandomEngine.h>
#include <CLHEP/Random/JamesRandom.h>

#include "GeneratorInterface/CosmicMuonGenerator/interface/CMSCGEN.h"
 
CMSCGEN::CMSCGEN() : initialization(0), RanGen2(0), delRanGen(false)
{
}

CMSCGEN::~CMSCGEN()
{
  if (delRanGen)
    delete RanGen2;
}

int CMSCGEN::initialize(double pmin_in, double pmax_in, double thetamin_in, double thetamax_in, CLHEP::HepRandomEngine *rnd, bool TIFOnly_constant, bool TIFOnly_linear)  
{
  if (delRanGen)
    delete RanGen2;
  RanGen2 = rnd;
  delRanGen = false;

  //set bools for TIFOnly options (E<2GeV with unphysical energy dependence)
  TIFOnly_const = TIFOnly_constant;
  TIFOnly_lin = TIFOnly_linear;


  // units: GeV

  // WARNING: coordinate system: 
  //   - to outside world define z axis downwards, i.e.
  //          muon coming from above, vertically: cos = 1
  //      (used for backward compatibility)
  //   - internally use frame with z axis upwards, i.e.
  //          muon coming from above, vertically: cos = -1
  //     (corresponds to CMS note definition)

  //set cmin and cmax, here convert between coordinate systems:
  cmin_in = - TMath::Cos(thetamin_in);//input angle already converted from Deg to Rad!
  cmax_in = - TMath::Cos(thetamax_in);//input angle already converted from Deg to Rad!


  //allowed energy range
  pmin_min = 3.;
  //pmin_max = 100.;
  pmin_max = 3000.;
  pmax = 3000.;
  //allowed angular range
  //cmax_max = -0.1,
  cmax_max = -0.01,
  cmax_min = -0.9999;

 if(TIFOnly_const == true || TIFOnly_lin == true) pmin_min = 0.; //forTIF

  // set pmin  
  if(pmin_in < pmin_min || pmin_in > pmin_max){ 
    std::cout << " >>> CMSCGEN.initialize <<< warning: illegal pmin_in =" << pmin_in;
    return(-1);
  } else if(pmax_in > pmax ){
    std::cout << " >>> CMSCGEN.initialize <<< warning: illegal pmax_in =" << pmax_in;
    return(-1);
  }else{     
    pmin = pmin_in;
    pmax = pmax_in;
    xemax = 1./(pmin*pmin);
    xemin = 1./(pmax*pmax); 
  }


  // set cmax and cmin
  if(cmax_in < cmax_min || cmax_in > cmax_max)
    { 
      std::cout << " >>> CMSCGEN.initialize <<< warning: illegal cmax_in =" << cmax_in;
      return(-1);
    }
  else 
    {
      cmax = cmax_in;
      cmin = cmin_in;
    }


  initialization = 1;

  if(TIFOnly_const == true || TIFOnly_lin == true) pmin_min = 3.; //forTIF

  //  Lmin = log10(pmin_min);
  Lmin = log10(pmin);
  Lmax = log10(pmax);
  Lfac = 100./(Lmax-Lmin);

  //
  // +++ coefficients for energy spectrum
  //

  pe[0] = -1.;
  pe[1] = 6.22176;
  pe[2] = -13.9404;
  pe[3] = 18.1643;
  pe[4] = -9.22784;
  pe[5] = 1.99234;
  pe[6] = -0.156434;
  pe[7] = 0.;
  pe[8] = 0.;

  //
  // +++ coefficients for cos theta distribution
  //

  b0c[0] = 0.6639;
  b0c[1] = -0.9587;
  b0c[2] = 0.2772;
  
  b1c[0] = 5.820;
  b1c[1] = -6.864;
  b1c[2] = 1.367;
  
  b2c[0] = 10.39;
  b2c[1] = -8.593;
  b2c[2] = 1.547;
  
  //
  // +++ calculate correction table for different cos theta dependence!
  //     reference range: c1  to  c2 
  //
  //  explanation: the parametrization of the energy spectrum as used above
  //    is the integral over the c = cos(zenith angle) range -1...-0.1 
  //    since the c distribution depends on energy, the integrated energy
  //    spectrum depends on this range. Here a correction factor is determined,
  //    based on the linear c dependence of the c distribution.
  //    The correction is calculated for 100 bins in L = log10(energy).
  //
  // +++ in same loop calculate integrated flux 
  //      (integrated over angles and momentum)

  c1 = -1.;
  c2 = -0.1;
  
  double cemax0 = 1.0;
  double L, L2; 
  double s;
  double p, p1, p2;
  double integral_here, integral_ref;
  double c_cut;

  integrated_flux = 0.;
  
  for(int k=1; k<=100; k++)
    {
      L = Lmin + (k-0.5)/Lfac;
      L2 = L*L;   
      p = pow(10,L);
      p1 = pow(10,L-0.5/Lfac);
      p2 = pow(10,L+0.5/Lfac);
      
      b0 = b0c[0] + b0c[1] * L + b0c[2]* L2;
      b1 = b1c[0] + b1c[1] * L + b1c[2]* L2;
      b2 = b2c[0] + b2c[1] * L + b2c[2]* L2;

      // cut out explicitly regions of zero flux 
      // (for low momentum and near horizontal showers)
      // since parametrization for z distribution doesn't work here
      // (can become negative) 
      
      c_cut = -0.42 + L*0.35;
      
      if (c_cut > c2) c_cut = c2; 
      
      integral_ref = b0 * (c_cut - c1) 
	+ b1/2. * (c_cut*c_cut - c1*c1) 
	+ b2/3. * (c_cut*c_cut*c_cut - c1*c1*c1);
      
      if (c_cut > cmax) c_cut = cmax; 
      
      integral_here = b0 * (c_cut - cmin) 
	+ b1/2. * (c_cut*c_cut - cmin*cmin) 
	+ b2/3. * (c_cut*c_cut*c_cut - cmin*cmin*cmin);
  
      corr[k] = integral_here/integral_ref;
      
      s = (((((((pe[8]*L
		 +pe[7])*L
		+pe[6])*L
	       +pe[5])*L
	      +pe[4])*L
	     +pe[3])*L
	    +pe[2])*L
	   +pe[1])*L
	+pe[0];
    
      integrated_flux += 1./pow(p,3) * s * corr[k] * (p2-p1);

      /*
	std::cout << k << " " 
	<< corr[k] << " " 
	<< p << " " 
	<< s << " " 
	<< p1 << " " 
	<< p2 << " " 
	<< integrated_flux << " " 
	<< std::endl;
      */
      
      // std::cout << k << " " << corr[k] << " " << std::endl;
    }

  integrated_flux *= 1.27E3;
  std::cout << " >>> CMSCGEN.initialize <<< " <<
    " Integrated flux = " << integrated_flux << " /m**2/s " << std::endl;
  
  //  find approximate peak value, for Monte Carlo sampling
  //      peak is near L = 2 

  double ce;
  
  ce = (((((((pe[8]*2.
	      +pe[7])*2.
	     +pe[6])*2.
            +pe[5])*2.
	   +pe[4])*2.
	  +pe[3])*2.
	 +pe[2])*2.
	+pe[1])*2.
    +pe[0];

  // normalize to 0.5 (not 1) to have some margin if peak is not at L=2
  //
  ce = 0.5/ce;

  for (int k=0; k<9; k++)
    {
      pe[k] = pe[k]*ce;
    }

  cemax = cemax0*corr[50];      

  return initialization;
}

int CMSCGEN::initialize(double pmin_in, double pmax_in, double thetamin_in, double thetamax_in, int RanSeed, bool TIFOnly_constant, bool TIFOnly_linear)
{
  CLHEP::HepRandomEngine *rnd = new CLHEP::HepJamesRandom;
  //set seed for Random Generator (seed can be controled by config-file), P.Biallass 2006
  rnd->setSeed(RanSeed, 0);
  delRanGen = true;
  return initialize(pmin_in, pmax_in, thetamin_in, thetamax_in, rnd, TIFOnly_constant, TIFOnly_linear);
}

int CMSCGEN::generate()
{

  if(initialization==0)
    {
      std::cout << " >>> CMSCGEN <<< warning: not initialized" << std::endl;
      return -1;
    }
  
  // note: use historical notation (fortran version l3cgen.f)

  //
  // +++ determine x = 1/e**2
  //
  //  explanation: the energy distribution 
  //        dn/d(1/e**2) = dn/de * e**3 = dn/dlog10(e) * e**2
  //     is parametrized by a polynomial. accordingly xe = 1/e**2 is sampled
  //     and e calculated 
  //
  //     need precise random variable with high precison since for 
  //     emin = 3 GeV energies around 3000 GeV are very rare!
  //

  double r1, r2, r3;
  double xe, e, ce, L, L2;
  int k;    
  double prob; 
  
  double c_max;
  double z, z_max;
  
  while (1)
    {
      prob = RanGen2->flat();
      r1 = double(prob);
      prob = RanGen2->flat();
      r2 = double(prob);
      
      xe = xemin+r1*(xemax-xemin);
      
      if( (1./sqrt(xe)<3) && TIFOnly_const == true) { //generate constant energy dependence for E<2GeV, only used for TIF
	//compute constant to match to CMSCGEN spectrum
	e=3.;      
	L = log10(e);
	L2 = L*L;
	
	ce = (((((((pe[8]*L
		    +pe[7])*L
		   +pe[6])*L
		  +pe[5])*L
		 +pe[4])*L
		+pe[3])*L
	       +pe[2])*L
	      +pe[1])*L
	  +pe[0];
	
	k = int ((L-Lmin)*Lfac+1.);
	k = TMath::Max(1,TMath::Min(k,100));
	ce = ce * corr[k];
	
	e = 1./sqrt(xe);  
	if(r2 < ( e*e*e*ce/(cemax*3.*3.*3.) )) break;
      
      }else if( (1./sqrt(xe)<3) && TIFOnly_lin == true) { //generate linear energy dependence for E<2GeV, only used for TIF
	//compute constant to match to CMSCGEN spectrum
	e=3.;      
	L = log10(e);
	L2 = L*L;
	
	ce = (((((((pe[8]*L
		    +pe[7])*L
		   +pe[6])*L
		  +pe[5])*L
		 +pe[4])*L
		+pe[3])*L
	       +pe[2])*L
	      +pe[1])*L
	  +pe[0];
      
	k = int ((L-Lmin)*Lfac+1.);
	k = TMath::Max(1,TMath::Min(k,100));
	ce = ce * corr[k];
      
	e = 1./sqrt(xe);  
	if(r2 < ( e*e*e*e*ce/(cemax*3.*3.*3.*3.) )) break;
      
      }else{ //this is real CMSCGEN energy-dependence

	e = 1./sqrt(xe);       
	L = log10(e);
	L2 = L*L;
	
	ce = (((((((pe[8]*L
		    +pe[7])*L
		   +pe[6])*L
		  +pe[5])*L
		 +pe[4])*L
		+pe[3])*L
	       +pe[2])*L
	      +pe[1])*L
	  +pe[0];
	
	k = int ((L-Lmin)*Lfac+1.);
	k = TMath::Max(1,TMath::Min(k,100));
	ce = ce * corr[k];
	
	if(cemax*r2 < ce) break;

      } //end of CMSCGEN energy-dependence
    } //end of while
  
  pq = e;
  
  //
  // +++ charge ratio 1.280
  //
  prob = RanGen2->flat();
  r3 = double(prob);
  
  double charg = 1.;
  if(r3 < 0.439) charg=-1.;
  
  pq = pq*charg;
  
  //
  //  +++ determine cos(angle)
  //
  //  simple trial and rejection method
  //
  
  // first calculate energy dependent coefficients b_i

  if(TIFOnly_const == true && e<3.){ //forTIF (when E<2GeV use angles of 2GeV cosmic)
    L = log10(3.);
    L2 = L*L;
  }
  if(TIFOnly_lin == true && e<3.){ //forTIF (when E<2GeV use angles of 2GeV cosmic)
    L = log10(3.);
    L2 = L*L; 
  }
  
  b0 = b0c[0] + b0c[1] * L + b0c[2]* L2;
  b1 = b1c[0] + b1c[1] * L + b1c[2]* L2;
  b2 = b2c[0] + b2c[1] * L + b2c[2]* L2;
  
  //
  // need to know the maximum of z(c)
  //
  // first calculate c for which c distribution z(c) = maximum 
  // 
  // (note: maximum of curve is NOT always at c = -1, but never at c = -0.1)   
  //
  
  // try extremal value (from z'(c) = 0), but only if z''(c) < 0  
  //
  // z'(c) = b1 + b2 * c   =>  at c_max = - b1 / (2 b_2) is z'(c) = 0
  //
  // z''(c) = b2 
  
  c_max = -1.;
  
  if(b2<0.) {
    c_max = - 0.5 * b1/b2;
    if(c_max < -1.) c_max = -1.;
    if(c_max > -0.1) c_max = -0.1;
  } 
  
  z_max = b0 + b1 * c_max + b2 * c_max * c_max;

  // again cut out explicitly regions of zero flux 
  double c_cut = -0.42 + L*0.35;
  if (c_cut > cmax) c_cut = cmax; 
  
  // now we throw dice:
  
  while (1)
    {
      prob = RanGen2->flat();
      r1 = double(prob);
      prob = RanGen2->flat();
      r2 = double(prob);
      c = cmin + (c_cut-cmin)*r1;    
      z = b0 + b1 * c + b2 * c*c;
      if (z > z_max*r2) break;
    }    
  
  return 0;
  
}


double CMSCGEN::momentum_times_charge()
{
  
  if(initialization==1)
    { 
      return pq;
    }
  else
    {
      std::cout << " >>> CMSCGEN <<< warning: not initialized" << std::endl;
      return -9999.;
    }

}

double CMSCGEN::cos_theta()
{
  
  if(initialization==1)
    { 
      // here convert between coordinate systems:
      return -c;
    }
  else
    {
      std::cout << " >>> CMSCGEN <<< warning: not initialized" << std::endl;
      return -0.9999;
    }

}

double CMSCGEN::flux()
{
  
  if(initialization==1)
    { 
      return integrated_flux;
    }
  else
    {
      std::cout << " >>> CMSCGEN <<< warning: not initialized" << std::endl;
      return -0.9999;
    }
  
}



int CMSCGEN::initializeNuMu(double pmin_in, double pmax_in, double thetamin_in, double thetamax_in, double Enumin_in, double Enumax_in, double Phimin_in, double Phimax_in, double ProdAlt_in, CLHEP::HepRandomEngine *rnd)
{
  if (delRanGen)
    delete RanGen2;
  RanGen2 = rnd;
  delRanGen = false;

  ProdAlt = ProdAlt_in;

  Rnunubar = 1.2;

  sigma = (0.72*Rnunubar+0.09)/(1+Rnunubar)*1.e-38; //cm^2GeV^-1

  AR = (0.69+0.06*Rnunubar)/(0.09+0.72*Rnunubar);


  //set smin and smax, here convert between coordinate systems:
  pmin = pmin_in;
  pmax = pmax_in;
  cmin = TMath::Cos(thetamin_in);//input angle already converted from Deg to Rad!
  cmax = TMath::Cos(thetamax_in);//input angle already converted from Deg to Rad!
  enumin = (Enumin_in < 10.) ? 10. : Enumin_in; //no nu's below 10GeV
  enumax = Enumax_in;


  //do initial run of flux rate to determine Maximum
  integrated_flux = 0.;
  dNdEmudEnuMax = 0.;
  negabs = 0.;
  negfrac = 0.;
  int trials = 100000;
  for (int i=0; i<trials; ++i) {
    double ctheta = cmin + (cmax-cmin)*RanGen2->flat();
    double Emu = pmin + (pmax-pmin)*RanGen2->flat();
    double Enu = enumin + (enumax-enumin)*RanGen2->flat();
    double rate =  dNdEmudEnu(Enu, Emu, ctheta);
    //std::cout << "trial=" << i << " ctheta=" << ctheta << " Emu=" << Emu << " Enu=" << Enu 
    //      << " rate=" << rate << std::endl;
    //std::cout << "cmin=" << cmin << " cmax=" << cmax 
    //      << " pmin=" << pmin << " pmax=" << pmax 
    //      << " enumin=" << enumin << " enumax=" << enumax << std::endl;
    if (rate > 0.) {
      integrated_flux += rate;
      if (rate > dNdEmudEnuMax)
	dNdEmudEnuMax = rate;
    }
    else negabs++;
  }
  negfrac = negabs/trials;
  integrated_flux /= trials;

  std::cout << "CMSCGEN::initializeNuMu: After " << trials << " trials:" << std::endl;
  std::cout << "dNdEmudEnuMax=" << dNdEmudEnuMax << std::endl;
  std::cout << "negfrac=" << negfrac << std::endl;

  //multiply by phase space boundaries
  integrated_flux *= (cmin-cmax);
  integrated_flux *= (Phimax_in-Phimin_in);
  integrated_flux *= (pmax-pmin);
  integrated_flux *= (enumax-enumin);
  //remove negative phase space areas which do not contribute anything
  integrated_flux *= (1.-negfrac);
  std::cout << " >>> CMSCGEN.initializeNuMu <<< " <<
    " Integrated flux = " << integrated_flux << " units??? " << std::endl;


  initialization = 1;

  return initialization;

} 

int CMSCGEN::initializeNuMu(double pmin_in, double pmax_in, double thetamin_in, double thetamax_in, double Enumin_in, double Enumax_in, double Phimin_in, double Phimax_in, double ProdAlt_in, int RanSeed)
{
  CLHEP::HepRandomEngine *rnd = new CLHEP::HepJamesRandom;
  //set seed for Random Generator (seed can be controled by config-file), P.Biallass 2006
  rnd->setSeed(RanSeed, 0);
  delRanGen = true;
  return initializeNuMu(pmin_in, pmax_in, thetamin_in, thetamax_in, Enumin_in, Enumax_in, Phimin_in, Phimax_in, ProdAlt_in, rnd);
}


double CMSCGEN::dNdEmudEnu(double Enu, double Emu, double ctheta) {
  double cthetaNu = 1. + ctheta; //swap cos(theta) from down to up range
  double thetas = asin(sin(acos(cthetaNu))*(Rearth-SurfaceOfEarth)/(Rearth+ProdAlt));
  double costhetas = cos(thetas);
  double dNdEnudW = 0.0286*pow(Enu,-2.7)*(1./(1.+(6.*Enu*costhetas)/115.)+0.213/(1.+(1.44*Enu*costhetas)/850.)); //cm^2*s*sr*GeV
  double dNdEmudEnu = N_A*sigma/alpha*dNdEnudW*1./(1.+Emu/epsilon)*
    (Enu-Emu+AR/3*(Enu*Enu*Enu-Emu*Emu*Emu)/(Enu*Enu));
  return dNdEmudEnu;
}


int CMSCGEN::generateNuMu() {
  if(initialization==0) 
    {
      std::cout << " >>> CMSCGEN <<< warning: not initialized" << std::endl;
      return -1;
    }
  
  double ctheta, Emu;
  while (1) {
    ctheta = cmin + (cmax-cmin)*RanGen2->flat();
    Emu = pmin + (pmax-pmin)*RanGen2->flat();
    double Enu = enumin + (enumax-enumin)*RanGen2->flat();
    double rate = dNdEmudEnu(Enu, Emu, ctheta);
    if (rate > dNdEmudEnuMax*RanGen2->flat()) break;
  }

  c = -ctheta; //historical sign convention

  pq = Emu;
  //
  // +++ nu/nubar ratio (~1.2)
  //
  double charg = 1.; //nubar -> mu+
  if (RanGen2->flat() > Rnunubar/(1.+Rnunubar))
    charg = -1.; //neutrino -> mu-

  pq = pq*charg;
 
  
  //int flux += this event rate

  return 1;
}
