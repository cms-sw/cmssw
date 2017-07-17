// #ifndef MT2_BISECT_C
// #define MT2_BISECT_C
/***********************************************************************/
/*                                                                     */
/*              Finding mt2 by Bisection                               */
/*                                                                     */
/*              Authors: Hsin-Chia Cheng, Zhenyu Han                   */ 
/*              Dec 11, 2008, v1.01a                                   */
/*                                                                     */  
/***********************************************************************/


/*******************************************************************************
Usage: 

1. Define an object of type "mt2":

mt2_bisect::mt2 mt2_event;

2. Set momenta and the mass of the invisible particle, mn:

mt2_event.set_momenta( pa, pb, pmiss );
mt2_event.set_mass( mn );

where array pa[0..2], pb[0..2], pmiss[0..2] contains (mass,px,py) 
	for the visible particles and the missing momentum. pmiss[0] is not used. 
	All quantities are given in double.    

	3. Use Davismt2::get_mt2() to obtain the value of mt2:

double mt2_value = mt2_event.get_mt2();       

*******************************************************************************/ 

#include "PhysicsTools/Heppy/interface/Davismt2.h"
// ClassImp(Davismt2);

using namespace std;

namespace heppy {

/*The user can change the desired precision below, the larger one of the following two definitions is used. Relative precision less than 0.00001 is not guaranteed to be 
  achievable--use with caution*/ 

const float Davismt2::RELATIVE_PRECISION = 0.00001;
const float Davismt2::ABSOLUTE_PRECISION = 0.0;
const float Davismt2::ZERO_MASS = 0.0; 
const float Davismt2::MIN_MASS = 0.1; 
const float Davismt2::SCANSTEP = 0.1;

Davismt2::Davismt2(){
	solved = false;
	momenta_set = false;
	mt2_b  = 0.;
	scale = 1.;
	verbose = 1;
}

Davismt2::~Davismt2(){}

double Davismt2::get_mt2(){
	if (!momenta_set)
	{
		cout << "Davismt2::get_mt2() ==> Please set momenta first!" << endl;
		return 0;
	}

	if (!solved) mt2_bisect();
	return mt2_b*scale;
}

void Davismt2::set_momenta(double* pa0, double* pb0, double* pmiss0){
	solved = false;     //reset solved tag when momenta are changed.
	momenta_set = true;

	ma = fabs(pa0[0]);  // mass cannot be negative

	if (ma < ZERO_MASS) ma = ZERO_MASS;

	pax  = pa0[1]; 
	pay  = pa0[2];
	masq = ma*ma;
	Easq = masq+pax*pax+pay*pay;
	Ea   = sqrt(Easq);

	mb = fabs(pb0[0]);

	if (mb < ZERO_MASS) mb = ZERO_MASS;

	pbx  = pb0[1]; 
	pby  = pb0[2];
	mbsq = mb*mb;
	Ebsq = mbsq+pbx*pbx+pby*pby;
	Eb   = sqrt(Ebsq);

	pmissx   = pmiss0[1]; pmissy = pmiss0[2];
	pmissxsq = pmissx*pmissx;
	pmissysq = pmissy*pmissy;

// set ma>= mb
	if(masq < mbsq)
	{
		double temp;
		temp = pax;  pax  = pbx;  pbx  = temp;
		temp = pay;  pay  = pby;  pby  = temp;
		temp = Ea;   Ea   = Eb;   Eb   = temp;
		temp = Easq; Easq = Ebsq; Ebsq = temp;
		temp = masq; masq = mbsq; mbsq = temp;
		temp = ma;   ma   = mb;   mb   = temp;   
	}
//normalize max{Ea, Eb} to 100
	if (Ea > Eb) scale = Ea/100.;
	else scale = Eb/100.;

	if (sqrt(pmissxsq+pmissysq)/100 > scale) scale = sqrt(pmissxsq+pmissysq)/100;
	//scale = 1;
	double scalesq = scale * scale;
	ma  = ma/scale;
	mb  = mb/scale;
	masq = masq/scalesq;
	mbsq = mbsq/scalesq;
	pax = pax/scale; pay = pay/scale;
	pbx = pbx/scale; pby = pby/scale;
	Ea  = Ea/scale;  Eb = Eb/scale;

	Easq = Easq/scalesq;
	Ebsq = Ebsq/scalesq;
	pmissx = pmissx/scale;
	pmissy = pmissy/scale;
	pmissxsq = pmissxsq/scalesq;
	pmissysq = pmissysq/scalesq;
	mn   = mn_unscale/scale; 
	mnsq = mn*mn;

	if (ABSOLUTE_PRECISION > 100.*RELATIVE_PRECISION) precision = ABSOLUTE_PRECISION;
	else precision = 100.*RELATIVE_PRECISION;
}

void Davismt2::set_mn(double mn0){
	solved = false;    //reset solved tag when mn is changed.
	mn_unscale   = fabs(mn0);  //mass cannot be negative
	mn = mn_unscale/scale;
	mnsq = mn*mn;
}

void Davismt2::print(){
	cout << " Davismt2::print() ==> pax = " << pax*scale << ";   pay = " << pay*scale << ";   ma = " << ma*scale <<";"<< endl;
	cout << " Davismt2::print() ==> pbx = " << pbx*scale << ";   pby = " << pby*scale << ";   mb = " << mb*scale <<";"<< endl;
	cout << " Davismt2::print() ==> pmissx = " << pmissx*scale << ";   pmissy = " << pmissy*scale <<";"<< endl;
	cout << " Davismt2::print() ==> mn = " << mn_unscale<<";" << endl;
}

//special case, the visible particle is massless
void Davismt2::mt2_massless(){

//rotate so that pay = 0 
	double theta,s,c;
	theta = atan(pay/pax);
	s = sin(theta);
	c = cos(theta);

	double pxtemp,pytemp;
	Easq   = pax*pax+pay*pay;
	Ebsq   = pbx*pbx+pby*pby;
	Ea     = sqrt(Easq);
	Eb     = sqrt(Ebsq);

	pxtemp = pax*c+pay*s;
	pax    = pxtemp;
	pay    = 0;
	pxtemp = pbx*c+pby*s;
	pytemp = -s*pbx+c*pby;
	pbx    = pxtemp;
	pby    = pytemp;
	pxtemp = pmissx*c+pmissy*s;
	pytemp = -s*pmissx+c*pmissy;
	pmissx = pxtemp;
	pmissy = pytemp;

	a2  = 1-pbx*pbx/(Ebsq);
	b2  = -pbx*pby/(Ebsq);
	c2  = 1-pby*pby/(Ebsq);

	d21 = (Easq*pbx)/Ebsq;
	d20 = - pmissx +  (pbx*(pbx*pmissx + pby*pmissy))/Ebsq;
	e21 = (Easq*pby)/Ebsq;
	e20 = - pmissy +  (pby*(pbx*pmissx + pby*pmissy))/Ebsq;
	f22 = -(Easq*Easq/Ebsq);
	f21 = -2*Easq*(pbx*pmissx + pby*pmissy)/Ebsq;
	f20 = mnsq + pmissxsq + pmissysq - (pbx*pmissx + pby*pmissy)*(pbx*pmissx + pby*pmissy)/Ebsq;

	double Deltasq0    = 0; 
	double Deltasq_low, Deltasq_high;
	int    nsols_high, nsols_low;

	Deltasq_low = Deltasq0 + precision;
	nsols_low = nsols_massless(Deltasq_low);

	if(nsols_low > 1) 
	{ 
		mt2_b = (double) sqrt(Deltasq0+mnsq);
		return;
	}

/*   
	if( nsols_massless(Deltasq_high) > 0 )
	{
		mt2_b = (double) sqrt(mnsq+Deltasq0);
		return;
		}*/

//look for when both parablos contain origin  
	double Deltasq_high1, Deltasq_high2;
	Deltasq_high1 = 2*Eb*sqrt(pmissx*pmissx+pmissy*pmissy+mnsq)-2*pbx*pmissx-2*pby*pmissy;
	Deltasq_high2 = 2*Ea*mn;

	if(Deltasq_high1 < Deltasq_high2) Deltasq_high = Deltasq_high2;
	else Deltasq_high = Deltasq_high1;

	nsols_high=nsols_massless(Deltasq_high);

	int foundhigh;
	if (nsols_high == nsols_low)
	{


		foundhigh=0;

		double minmass, maxmass;
		minmass  = mn ;
		maxmass  = sqrt(mnsq + Deltasq_high);
		for(double mass = minmass + SCANSTEP; mass < maxmass; mass += SCANSTEP)
		{
			Deltasq_high = mass*mass - mnsq;

			nsols_high = nsols_massless(Deltasq_high);
			if(nsols_high>0)
			{
				foundhigh=1;
				Deltasq_low = (mass-SCANSTEP)*(mass-SCANSTEP) - mnsq;
				break;
			}
		}
		if(foundhigh==0) 
		{

			if(verbose > 0) cout << "Davismt2::mt2_massless() ==> Deltasq_high not found at event " << nevt <<endl;


			mt2_b = (double)sqrt(Deltasq_low+mnsq);
			return;
		}
	}

	if(nsols_high == nsols_low)
	{ 
		if(verbose > 0) cout << "Davismt2::mt2_massless() ==> error: nsols_low=nsols_high=" << nsols_high << endl;
		if(verbose > 0) cout << "Davismt2::mt2_massless() ==> Deltasq_high=" << Deltasq_high << endl;
		if(verbose > 0) cout << "Davismt2::mt2_massless() ==> Deltasq_low= "<< Deltasq_low << endl;

		mt2_b = sqrt(mnsq + Deltasq_low);
		return;
	}
	double minmass, maxmass;
	minmass = sqrt(Deltasq_low+mnsq);
	maxmass = sqrt(Deltasq_high+mnsq);
	while(maxmass - minmass > precision)
	{
		double Delta_mid, midmass, nsols_mid;
		midmass   = (minmass+maxmass)/2.;
		Delta_mid = midmass * midmass - mnsq;
		nsols_mid = nsols_massless(Delta_mid);
		if(nsols_mid != nsols_low) maxmass = midmass;
		if(nsols_mid == nsols_low) minmass = midmass;
	}
	mt2_b = minmass;
	return;
}

int Davismt2::nsols_massless(double Dsq){
	double delta;
	delta = Dsq/(2*Easq);
	d1    = d11*delta;
	e1    = e11*delta;
	f1    = f12*delta*delta+f10;
	d2    = d21*delta+d20;
	e2    = e21*delta+e20;
	f2    = f22*delta*delta+f21*delta+f20;

	double a,b;
	if (pax > 0) a = Ea/Dsq;
	else         a = -Ea/Dsq;
	if (pax > 0) b = -Dsq/(4*Ea)+mnsq*Ea/Dsq;
	else         b = Dsq/(4*Ea)-mnsq*Ea/Dsq;

	double A4,A3,A2,A1,A0;

	A4 = a*a*a2;
	A3 = 2*a*b2/Ea;
	A2 = (2*a*a2*b+c2+2*a*d2)/(Easq);
	A1 = (2*b*b2+2*e2)/(Easq*Ea);
	A0 = (a2*b*b+2*b*d2+f2)/(Easq*Easq);

        long double A3sq;
        A3sq = A3*A3;
        /*	
        long  double A0sq, A1sq, A2sq, A3sq, A4sq;
	A0sq = A0*A0;
	A1sq = A1*A1;
	A2sq = A2*A2;
	A3sq = A3*A3;
	A4sq = A4*A4;
        */

	long double B3, B2, B1, B0;
	B3 = 4*A4;
	B2 = 3*A3;
	B1 = 2*A2;
	B0 = A1;
	long double C2, C1, C0;
	C2 = -(A2/2 - 3*A3sq/(16*A4));
	C1 = -(3*A1/4. -A2*A3/(8*A4));
	C0 = -A0 + A1*A3/(16*A4);
	long double  D1, D0;
	D1 = -B1 - (B3*C1*C1/C2 - B3*C0 -B2*C1)/C2;
	D0 = -B0 - B3 *C0 *C1/(C2*C2)+ B2*C0/C2;

	long double E0;
	E0 = -C0 - C2*D0*D0/(D1*D1) + C1*D0/D1;

	long  double t1,t2,t3,t4,t5;

//find the coefficients for the leading term in the Sturm sequence  
	t1 = A4;
	t2 = A4;
	t3 = C2;
	t4 = D1;
	t5 = E0;

	int nsol;
	nsol = signchange_n(t1,t2,t3,t4,t5)-signchange_p(t1,t2,t3,t4,t5);
	if( nsol < 0 ) nsol=0;

	return nsol;
}

void Davismt2::mt2_bisect(){

	solved = true;
	cout.precision(11);

//if masses are very small, use code for massless case.  
	if(masq < MIN_MASS && mbsq < MIN_MASS){ 
		mt2_massless();
		return;
	}


	double Deltasq0;     
	Deltasq0 = ma*(ma + 2*mn); //The minimum mass square to have two ellipses 

// find the coefficients for the two quadratic equations when Deltasq=Deltasq0.

	a1 = 1-pax*pax/(Easq);
	b1 = -pax*pay/(Easq);
	c1 = 1-pay*pay/(Easq);
	d1 = -pax*(Deltasq0-masq)/(2*Easq);
	e1 = -pay*(Deltasq0-masq)/(2*Easq);
	a2 = 1-pbx*pbx/(Ebsq);
	b2 = -pbx*pby/(Ebsq);
	c2 = 1-pby*pby/(Ebsq);
	d2 = -pmissx+pbx*(Deltasq0-mbsq)/(2*Ebsq)+pbx*(pbx*pmissx+pby*pmissy)/(Ebsq);
	e2 = -pmissy+pby*(Deltasq0-mbsq)/(2*Ebsq)+pby*(pbx*pmissx+pby*pmissy)/(Ebsq);
	f2 = pmissx*pmissx+pmissy*pmissy-((Deltasq0-mbsq)/(2*Eb)+
		(pbx*pmissx+pby*pmissy)/Eb)*((Deltasq0-mbsq)/(2*Eb)+
		(pbx*pmissx+pby*pmissy)/Eb)+mnsq;

// find the center of the smaller ellipse 
	double x0,y0;
	x0 = (c1*d1-b1*e1)/(b1*b1-a1*c1);
	y0 = (a1*e1-b1*d1)/(b1*b1-a1*c1);


// Does the larger ellipse contain the smaller one? 
	double dis=a2*x0*x0+2*b2*x0*y0+c2*y0*y0+2*d2*x0+2*e2*y0+f2;

	if(dis<=0.01)
	{ 
		mt2_b  = (double) sqrt(mnsq+Deltasq0);
		return;
	}


/* find the coefficients for the two quadratic equations           */
/* coefficients for quadratic terms do not change                  */
/* coefficients for linear and constant terms are polynomials of   */
/*       delta=(Deltasq-m7sq)/(2 E7sq)                             */  
	d11 = -pax;
	e11 = -pay;
	f10 = mnsq;
	f12 = -Easq;
	d21 = (Easq*pbx)/Ebsq;
	d20 = ((masq - mbsq)*pbx)/(2.*Ebsq) - pmissx +
		(pbx*(pbx*pmissx + pby*pmissy))/Ebsq;
	e21 = (Easq*pby)/Ebsq;
	e20 = ((masq - mbsq)*pby)/(2.*Ebsq) - pmissy +
		(pby*(pbx*pmissx + pby*pmissy))/Ebsq;
	f22 = -Easq*Easq/Ebsq;
	f21 = (-2*Easq*((masq - mbsq)/(2.*Eb) + (pbx*pmissx + pby*pmissy)/Eb))/Eb;
	f20 = mnsq + pmissx*pmissx + pmissy*pmissy - 
		((masq - mbsq)/(2.*Eb) + (pbx*pmissx + pby*pmissy)/Eb)
		*((masq - mbsq)/(2.*Eb) + (pbx*pmissx + pby*pmissy)/Eb);

//Estimate upper bound of mT2
//when Deltasq > Deltasq_high1, the larger encloses the center of the smaller 
	double p2x0,p2y0;
	double Deltasq_high1;
	p2x0 = pmissx-x0;
	p2y0 = pmissy-y0;
	Deltasq_high1 = 2*Eb*sqrt(p2x0*p2x0+p2y0*p2y0+mnsq)-2*pbx*p2x0-2*pby*p2y0+mbsq;

//Another estimate, if both ellipses enclose the origin, Deltasq > mT2

	double Deltasq_high2, Deltasq_high21, Deltasq_high22;
	Deltasq_high21 = 2*Eb*sqrt(pmissx*pmissx+pmissy*pmissy+mnsq)-2*pbx*pmissx-2*pby*pmissy+mbsq;
	Deltasq_high22 = 2*Ea*mn+masq;

	if ( Deltasq_high21 < Deltasq_high22 ) Deltasq_high2 = Deltasq_high22;
	else Deltasq_high2 = Deltasq_high21;

//pick the smaller upper bound   
	double Deltasq_high;
	if(Deltasq_high1 < Deltasq_high2) Deltasq_high = Deltasq_high1;
	else Deltasq_high = Deltasq_high2;


	double Deltasq_low; //lower bound
	Deltasq_low = Deltasq0;

//number of solutions at Deltasq_low should not be larger than zero
	if( nsols(Deltasq_low) > 0 )
	{
//cout << "Davismt2::mt2_bisect() ==> nsolutions(Deltasq_low) > 0"<<endl;
		mt2_b = (double) sqrt(mnsq+Deltasq0);
		return;
	}

	int nsols_high, nsols_low;

	nsols_low  = nsols(Deltasq_low);
	int foundhigh;


//if nsols_high=nsols_low, we missed the region where the two ellipse overlap 
//if nsols_high=4, also need a scan because we may find the wrong tangent point.

	nsols_high = nsols(Deltasq_high);

	if(nsols_high == nsols_low || nsols_high == 4)
	{
	//foundhigh = scan_high(Deltasq_high);
		foundhigh = find_high(Deltasq_high);
		if(foundhigh == 0) 
		{
			if(verbose > 0) cout << "Davismt2::mt2_bisect() ==> Deltasq_high not found at event " << nevt << endl;
			mt2_b = sqrt( Deltasq_low + mnsq );
			return;
		}

	}

	while(sqrt(Deltasq_high+mnsq) - sqrt(Deltasq_low+mnsq) > precision)
	{
		double Deltasq_mid,nsols_mid;
	//bisect
		Deltasq_mid = (Deltasq_high+Deltasq_low)/2.;
		nsols_mid = nsols(Deltasq_mid);
	// if nsols_mid = 4, rescan for Deltasq_high
		if ( nsols_mid == 4 ) 
		{
			Deltasq_high = Deltasq_mid;
		//scan_high(Deltasq_high);
			find_high(Deltasq_high);
			continue;
		} 


		if(nsols_mid != nsols_low) Deltasq_high = Deltasq_mid;
		if(nsols_mid == nsols_low) Deltasq_low  = Deltasq_mid;
	}
	mt2_b = (double) sqrt( mnsq + Deltasq_high);
	return;
}

int Davismt2::find_high(double & Deltasq_high){
	double x0,y0;
	x0 = (c1*d1-b1*e1)/(b1*b1-a1*c1);
	y0 = (a1*e1-b1*d1)/(b1*b1-a1*c1);
	double Deltasq_low = (mn + ma)*(mn + ma) - mnsq;
	do 
	{
		double Deltasq_mid = (Deltasq_high + Deltasq_low)/2.;
		int nsols_mid = nsols(Deltasq_mid);
		if ( nsols_mid == 2 )
		{
			Deltasq_high = Deltasq_mid;
			return 1;
		}
		else if (nsols_mid == 4)
		{
			Deltasq_high = Deltasq_mid;
			continue;
		}
		else if (nsols_mid ==0)
		{
			d1 = -pax*(Deltasq_mid-masq)/(2*Easq);
			e1 = -pay*(Deltasq_mid-masq)/(2*Easq);
			d2 = -pmissx + pbx*(Deltasq_mid - mbsq)/(2*Ebsq)
				+ pbx*(pbx*pmissx+pby*pmissy)/(Ebsq);
			e2 = -pmissy + pby*(Deltasq_mid - mbsq)/(2*Ebsq)
				+ pby*(pbx*pmissx+pby*pmissy)/(Ebsq);
			f2 = pmissx*pmissx+pmissy*pmissy-((Deltasq_mid-mbsq)/(2*Eb)+
				(pbx*pmissx+pby*pmissy)/Eb)*((Deltasq_mid-mbsq)/(2*Eb)+
				(pbx*pmissx+pby*pmissy)/Eb)+mnsq;
// Does the larger ellipse contain the smaller one? 
			double dis = a2*x0*x0 + 2*b2*x0*y0 + c2*y0*y0 + 2*d2*x0 + 2*e2*y0 + f2;
			if (dis < 0) Deltasq_high = Deltasq_mid;
			else Deltasq_low = Deltasq_mid;
		}

	} while ( Deltasq_high - Deltasq_low > 0.001);
	return 0;
}

int Davismt2::scan_high(double & Deltasq_high){
	int foundhigh = 0 ;
	int nsols_high;


        //        double Deltasq_low;
	double tempmass, maxmass;
	tempmass = mn + ma;
	maxmass  = sqrt(mnsq + Deltasq_high);
	if (nevt == 32334) cout << "Davismt2::scan_high() ==> Deltasq_high = " << Deltasq_high << endl; // ???
	for(double mass = tempmass + SCANSTEP; mass < maxmass; mass += SCANSTEP)
	{
		Deltasq_high = mass*mass - mnsq;
		nsols_high   = nsols(Deltasq_high);

		if( nsols_high > 0)
		{
                  // Deltasq_low = (mass-SCANSTEP)*(mass-SCANSTEP) - mnsq;
			foundhigh   = 1;
			break;
		}
	}
	return foundhigh;
}

int Davismt2::nsols(double Dsq){
	double delta = (Dsq-masq)/(2*Easq);

//calculate coefficients for the two quadratic equations
	d1 = d11*delta;
	e1 = e11*delta;
	f1 = f12*delta*delta+f10;
	d2 = d21*delta+d20;
	e2 = e21*delta+e20;
	f2 = f22*delta*delta+f21*delta+f20;

//obtain the coefficients for the 4th order equation 
//devided by Ea^n to make the variable dimensionless
	long double A4, A3, A2, A1, A0;

	A4 = 
		-4*a2*b1*b2*c1 + 4*a1*b2*b2*c1 +a2*a2*c1*c1 + 
		4*a2*b1*b1*c2 - 4*a1*b1*b2*c2 - 2*a1*a2*c1*c2 + 
		a1*a1*c2*c2;  

	A3 =
		(-4*a2*b2*c1*d1 + 8*a2*b1*c2*d1 - 4*a1*b2*c2*d1 - 4*a2*b1*c1*d2 + 
		8*a1*b2*c1*d2 - 4*a1*b1*c2*d2 - 8*a2*b1*b2*e1 + 8*a1*b2*b2*e1 + 
		4*a2*a2*c1*e1 - 4*a1*a2*c2*e1 + 8*a2*b1*b1*e2 - 8*a1*b1*b2*e2 - 
		4*a1*a2*c1*e2 + 4*a1*a1*c2*e2)/Ea;


	A2 =
		(4*a2*c2*d1*d1 - 4*a2*c1*d1*d2 - 4*a1*c2*d1*d2 + 4*a1*c1*d2*d2 - 
		8*a2*b2*d1*e1 - 8*a2*b1*d2*e1 + 16*a1*b2*d2*e1 + 
		4*a2*a2*e1*e1 + 16*a2*b1*d1*e2 - 8*a1*b2*d1*e2 - 
		8*a1*b1*d2*e2 - 8*a1*a2*e1*e2 + 4*a1*a1*e2*e2 - 4*a2*b1*b2*f1 + 
		4*a1*b2*b2*f1 + 2*a2*a2*c1*f1 - 2*a1*a2*c2*f1 + 
		4*a2*b1*b1*f2 - 4*a1*b1*b2*f2 - 2*a1*a2*c1*f2 + 2*a1*a1*c2*f2)/Easq;

	A1 =
		(-8*a2*d1*d2*e1 + 8*a1*d2*d2*e1 + 8*a2*d1*d1*e2 - 8*a1*d1*d2*e2 - 
		4*a2*b2*d1*f1 - 4*a2*b1*d2*f1 + 8*a1*b2*d2*f1 + 4*a2*a2*e1*f1 - 
		4*a1*a2*e2*f1 + 8*a2*b1*d1*f2 - 4*a1*b2*d1*f2 - 4*a1*b1*d2*f2 - 
		4*a1*a2*e1*f2 + 4*a1*a1*e2*f2)/(Easq*Ea);

	A0 =
		(-4*a2*d1*d2*f1 + 4*a1*d2*d2*f1 + a2*a2*f1*f1 + 
		4*a2*d1*d1*f2 - 4*a1*d1*d2*f2 - 2*a1*a2*f1*f2 + 
		a1*a1*f2*f2)/(Easq*Easq);

        long double A3sq;
        /*
	long  double A0sq, A1sq, A2sq, A3sq, A4sq;
	A0sq = A0*A0;
	A1sq = A1*A1;
	A2sq = A2*A2;
	A4sq = A4*A4;
        */
	A3sq = A3*A3;

	long double B3, B2, B1, B0;
	B3 = 4*A4;
	B2 = 3*A3;
	B1 = 2*A2;
	B0 = A1;

	long double C2, C1, C0;
	C2 = -(A2/2 - 3*A3sq/(16*A4));
	C1 = -(3*A1/4. -A2*A3/(8*A4));
	C0 = -A0 + A1*A3/(16*A4);

	long double D1, D0;
	D1 = -B1 - (B3*C1*C1/C2 - B3*C0 -B2*C1)/C2;
	D0 = -B0 - B3 *C0 *C1/(C2*C2)+ B2*C0/C2;

	long double E0;
	E0 = -C0 - C2*D0*D0/(D1*D1) + C1*D0/D1;

	long  double t1,t2,t3,t4,t5;
//find the coefficients for the leading term in the Sturm sequence  
	t1 = A4;
	t2 = A4;
	t3 = C2;
	t4 = D1;
	t5 = E0;


//The number of solutions depends on diffence of number of sign changes for x->Inf and x->-Inf
	int nsol;
	nsol = signchange_n(t1,t2,t3,t4,t5) - signchange_p(t1,t2,t3,t4,t5);

//Cannot have negative number of solutions, must be roundoff effect
	if (nsol < 0) nsol = 0;

	return nsol;

}  

//inline
int Davismt2::signchange_n(long double t1, long double t2, long double t3, long double t4, long double t5){
	int nsc;
	nsc=0;
	if(t1*t2>0) nsc++;
	if(t2*t3>0) nsc++;
	if(t3*t4>0) nsc++;
	if(t4*t5>0) nsc++;
	return nsc;
}

//inline
int Davismt2::signchange_p(long double t1, long double t2, long double t3, long double t4, long double t5){
	int nsc;
	nsc=0;
	if(t1*t2<0) nsc++;
	if(t2*t3<0) nsc++;
	if(t3*t4<0) nsc++;
	if(t4*t5<0) nsc++;
	return nsc;
}

}

