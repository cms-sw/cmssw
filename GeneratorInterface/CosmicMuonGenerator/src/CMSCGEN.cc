//
// CMSCGEN.cc       T.Hebbeker 2006
//
// implemented in CMSSW by P. Biallass 29.03.2006  
// see header for documentation
//


#include "GeneratorInterface/CosmicMuonGenerator/interface/CMSCGEN.h"


int CMSCGEN::initialize(float Emin_in, float Emax_in, float thetamin_in, float thetamax_in, int RanSeed, bool TIFOnly_constant, bool TIFOnly_linear) 
{
  //set seed for Random Generator (seed can be controled by config-file), P.Biallass 2006
  RanGen2.SetSeed(RanSeed);

  //set bools for TIFOnly options (E<2GeV with unphysical energy dependence)
  TIFOnly_const = TIFOnly_constant;
  TIFOnly_lin = TIFOnly_linear;

  // units: GeV

  // muon coming from above, vertically: cos = 1

  //allowed energy range
  Emin_min=2.;
  Emin_max=1000.;
  Emax_max=10000.;   
  //allowed angular range
  cmin_min=0.0348994;
  cmin_max=0.9999;
  cmax = 1.;

  // set Emin  

  if(TIFOnly_constant == true || TIFOnly_linear == true) Emin_min = 0.; //forTIF

  if(Emin_in < Emin_min || Emin_in > Emin_max)
  { 
    std::cout << " >>> CMSCGEN.initialize <<< warning: illegal Emin_in =" << Emin_in;
    return(-1);
  }
  else if(Emax_in > Emax_max )
  {
   std::cout << " >>> CMSCGEN.initialize <<< warning: illegal Emax_in =" << Emax_in;
    return(-1);
  }else{   
    Emin = Emin_in;
    Emax = Emax_in;
    xemax = 1./(Emin*Emin);
    xemin = 1./(Emax*Emax); 
  }

  //set cmin and cmax
 if( TMath::Cos(thetamin_in) < cmax )
  { 
    std::cout << " >>> CMSCGEN.initialize <<< warning: illegal thetamin_in =" << thetamin_in;
    return(-1);
  }
  else if(TMath::Cos(thetamax_in) < cmin_min || TMath::Cos(thetamax_in) > cmin_max)
  {
    std::cout << " >>> CMSCGEN.initialize <<< warning: illegal thetamax_in =" << thetamax_in;
    return(-1);
  }else{   
    cmin = TMath::Cos(thetamax_in);//input angle already converted from Deg to Rad!
    cmax = TMath::Cos(thetamin_in);//input angle already converted from Deg to Rad!
  }

  initialization = 1;

  if(TIFOnly_constant == true || TIFOnly_linear == true) Emin_min = 2.; //forTIF
  elmin = log10(Emin_min);
  elmax = log10(Emax_max);
  elfac = 100./(elmax-elmin);

//
// +++ calculate coefficients for energy spectrum
//

  pe[0] = -1.472;
  pe[1] = 14.42;
  pe[2] = -47.92;
  pe[3] = 80.94;
  pe[4] = -62.84;
  pe[5] = 25.95;
  pe[6] = -6.014;
  pe[7] = 0.7456;
  pe[8] = -0.0387;

//  normalisation !  Note: 2 = log10(100)   

  float ce;

  ce = (((((((pe[8]*2.
            +pe[7])*2.
            +pe[6])*2.
            +pe[5])*2.
            +pe[4])*2.
            +pe[3])*2.
            +pe[2])*2.
            +pe[1])*2.
            +pe[0];

  ce = 0.28/ce;

  for (int k=0; k<9; k++)
  {
    pe[k] = pe[k]*ce;
  }

  pc[0] = -1.903;
  pc[1] = 0.1434;
  pc[2] = 0.01450;

//
// +++ calculate correction table for different cos theta dependence!
//     reference range: c1  to  c2 
//
//  explanation: the parametrization of the energy spectrum as used above
//    is the integral over the c = cos(zenith angle) range 0.4-1
//    since the c distribution depends on energy, the integrated energy
//    spectrum depends on this range. Here a correction factor is determined,
//    based on the linear c dependence of the c distribution.
//    The correction is calculated for 100 bins in log10(energy).
//

  c1 = 0.4;
  c2 = 1.;

  float cemax0 = 0.3;
  float e10, ee, cax, ca, here, ref;

  for(int k=1; k<=100; k++)
  {
    e10 = elmin + (elmax-elmin)/100.*(k-0.5);
    ee = e10*log(10.);
    cax = (pc[2]*ee+pc[1])*ee+pc[0];
    ca = -cax/(1.+cax);

    // Integration for a specific energy should be done from theta_min to 
    // angle where flux becomes zero (= cmin_allowed).
    // Decide here if this angle is lower than theta_max defined earlier.
    // Large energies lead to unphysical cmin_allowed (if cax>0), choose cmin in this case
    // as high energies can have very large angles, so flux never zero.
    cmin_allowed = 0.;
    if(cax < 0.) cmin_allowed = 1 + 1./cax;
    cmin_allowed = TMath::Max(cmin, cmin_allowed);

    // TH  version 1.1   following two lines were wrong in version 1.0
    //   ( instead of   +0.5*ca  there was written   *0.5*ca  )
    ref = (c2-c1)+0.5*ca*(c2*c2-c1*c1);
    here = (cmax-cmin_allowed)+0.5*ca*(cmax*cmax-cmin_allowed*cmin_allowed);
    // 
    corr[k] = here/ref;
  }
   
  cemax = cemax0*corr[50];      

  return initialization;
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
//     emin = 2 GeV energies around 10000 GeV are very rare!
//     [roughly (2/10000)**3 = 8E-12]
//
  
  bool accept = 0;
  float r1, r2, r3, r4;
  float xe, e, ce, e10;
  int k;    
  double prob; 

  while (accept==0)
    {
      
      // P. Biallass, 2006:
      prob = RanGen2.Rndm();
      r1 = float(prob);
      prob = RanGen2.Rndm();
      r2 = float(prob);
      
      xe = xemin+r1*(xemax-xemin);
      if( (1./sqrt(xe)<2) && TIFOnly_const == true) { //generate constant energy dependence for E<2GeV, only used for TIF
	//compute constant to match to CMSCGEN spectrum
	e=2.;      
	e10 = log10(e);
	
	ce = (((((((pe[8]*e10
		    +pe[7])*e10
		   +pe[6])*e10
		  +pe[5])*e10
		 +pe[4])*e10
		+pe[3])*e10
	       +pe[2])*e10
	      +pe[1])*e10
	  +pe[0];
	
	k = int ((e10-elmin)*elfac+1.);
	k = TMath::Max(1,TMath::Min(k,100));
	ce = ce * corr[k];
	
	e = 1./sqrt(xe);  
	if(r2 < ( e*e*e*ce/(cemax*2.*2.*2.) ))
	  {
	    accept = 1;
	  }
      }else if( (1./sqrt(xe)<2) && TIFOnly_lin == true) { //generate linear energy dependence for E<2GeV, only used for TIF
	//compute constant to match to CMSCGEN spectrum
	e=2.;      
	e10 = log10(e);
	
	ce = (((((((pe[8]*e10
		    +pe[7])*e10
		   +pe[6])*e10
		  +pe[5])*e10
		 +pe[4])*e10
		+pe[3])*e10
	       +pe[2])*e10
	      +pe[1])*e10
	  +pe[0];
      
	k = int ((e10-elmin)*elfac+1.);
	k = TMath::Max(1,TMath::Min(k,100));
	ce = ce * corr[k];
      
	e = 1./sqrt(xe);  
	if(r2 < ( e*e*e*e*ce/(cemax*2.*2.*2.*2.) ))
	  {
	    accept = 1;
	  }
      }else{ //this is real CMSCGEN energy-dependence
	e = 1./sqrt(xe);       
	e10 = log10(e);
	
	ce = (((((((pe[8]*e10
		    +pe[7])*e10
		   +pe[6])*e10
		  +pe[5])*e10
		 +pe[4])*e10
		+pe[3])*e10
	       +pe[2])*e10
	      +pe[1])*e10
	  +pe[0];
	
	k = int ((e10-elmin)*elfac+1.);
	k = TMath::Max(1,TMath::Min(k,100));
	ce = ce * corr[k];
      
	if(cemax*r2 < ce)
	  {
	    accept = 1;
	  }
      } //end of CMSCGEN energy-dependence
    } //end of while
     
  pq = e;

//
// +++ charge ratio 1.3
//
  prob = RanGen2.Rndm();
  r3 = float(prob);

  float charg = 1.;
  if(r3 < 0.43) charg=-1.;
       
  pq = pq*charg;

//
//  +++ determine cos(angle)
//
//     here we can calculate c analytically, since cos(angle) = c
//     distribution is linear (to a good approximation)
//      ca is linear coefficient in c distribution:
//     dN/dc = 1 + ca * c
//      integration:  int ~ (c-cmin) + 0.5*ca*(c**2-cmin**2)
//     normalization:  factor 1/[(cmax-cmin)+0.5*ca*(cmax**2-cmin**2)]
//     inversion: int = r   yields  solution of equation c**2 + pc + q = 0:
//
  
  float eln, cax, fac, p, q, ca; 

  if(TIFOnly_const == true && e<2.) e = 2.; //forTIF (when E<2GeV use angles of 2GeV cosmic)
  if(TIFOnly_lin == true && e<2.) e = 2.; //forTIF (when E<2GeV use angles of 2GeV cosmic)

  eln = log(e);
  cax = (pc[2]*eln+pc[1])*eln+pc[0];

  if (fabs(1.+cax) < 0.001) 
  {
    ca = -1000.;
  } 
  else 
  {       
    ca = -cax/(1.+cax);
  }

// Volker Schmitt:
  if(fabs(ca) < 0.001) 
  {
    if (ca < 0) ca=0.001;
    if (ca >=0) ca=-0.001;
// TH march 1999:
    cax = -ca;
  }

  // Integration for a specific energy should be done from theta_min to 
  // angle where flux becomes zero (= cmin_allowed).
  // Decide here if this angle is lower than theta_max defined earlier.
  // Large energies lead to unphysical cmin_allowed (if cax>0), choose cmin in this case
  // as high energies can have very large angles, so flux never zero.
  cmin_allowed = 0.;
  if(cax < 0.) cmin_allowed = 1 + 1./cax;
  cmin_allowed = TMath::Max(cmin, cmin_allowed);

  //finally dice angle    
  prob = RanGen2.Rndm();
  r4 = float(prob);
  fac = (cmax-cmin_allowed)+0.5*ca*(cmax*cmax-cmin_allowed*cmin_allowed);
  p = 2./ca;
  q = -p*cmin_allowed - cmin_allowed*cmin_allowed - p*fac*r4;
  if(cax >= 0.) c = -0.5*p - sqrt(0.25*p*p-q);
  if(cax < 0.) c = -0.5*p + sqrt(0.25*p*p-q);

  if(c>cmax || c<cmin) std::cout << " >>> CMSCGEN <<< warning: generated cosmic with cos(theta) = " << c << " --> out of range "<< std::endl;

  return 0;

}


float CMSCGEN::energy_times_charge()
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

float CMSCGEN::cos_theta()
{

  if(initialization==1)
  { 
    return c;
  }
  else
  {
    std::cout << " >>> CMSCGEN <<< warning: not initialized" << std::endl;
    return -0.9999;
  }

}


