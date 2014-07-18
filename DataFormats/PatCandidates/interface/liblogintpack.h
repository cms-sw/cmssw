#ifndef liblogintpack_h
#define liblogintpack_h

#include <cmath>

namespace logintpack
{
	int8_t pack8log(double x,double lmin, double lmax, uint8_t base=128)
	{
	        if(base>128) base=128;
		float l =log(fabs(x));
		float centered = (l-lmin)/(lmax-lmin)*base;
		int8_t  r=centered;
		if(centered >= base-1) return r=base-1;
		if(centered < 0) return r=0;
		if(x<0) r=-r;
		return r;
	}

	double unpack8log(int8_t i,double lmin, double lmax, uint8_t base=128)
	{
	        if(base>128) base=128;
	        float basef=base;
		float l=lmin+abs(i)/basef*(lmax-lmin);
		float val=exp(l);
		if(i<0) return -val; else return val;
	}
}
#endif
