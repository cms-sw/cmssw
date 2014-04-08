#ifndef liblogintpack_h
#define liblogintpack_h

#include <cmath>

namespace logintpack
{
	int8_t pack8log(double x,double lmin, double lmax)
	{
		float l =log(fabs(x));
		float centered = (l-lmin)/(lmax-lmin)*128;
		int8_t  r=centered;
		if(centered >= 127) return r=127;
		if(centered < 0) return r=0;
		if(x<0) r=-r;
		return r;
	}

	double unpack8log(int8_t i,double lmin, double lmax)
	{
		float l=lmin+abs(i)/128.*(lmax-lmin);
		float val=exp(l);
		if(i<0) return -val; else return val;
	}
}
#endif
