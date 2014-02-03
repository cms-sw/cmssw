//FAMOS headers
#include "FastSimulation/Utilities/interface/DoubleCrystalBallGenerator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

//ROOT headers
#include "TMath.h"
#include <iostream>

using namespace TMath;

double DoubleCrystalBallGenerator::shoot(double mu, double sigma, double aL, double nL, double aR, double nR){
	if(nL<=1 || nR<=1) return 0; //n>1 required
	
	double dL = nL/aL;
	double dR = nR/aR;
	double N = 1/(sigma*(nL/aL*1/(nL-1)*Exp(-aL*aL/2) + Sqrt(Pi()/2)*(Erf(aL/Sqrt(2))+Erf(aR/Sqrt(2))) + nR/aR*1/(nR-1)*Exp(-aR*aR/2)));

	//start x as NaN
	double x = 0./0.;
	while(!Finite(x)){
		//shoot a flat random number
		double y = random->flatShoot();
		
		//check range
		//crystal ball CDF changes at (x-mu)/sigma = -aL && (x-mu)/sigma = aR
		//compute this in pieces because it is awful
		double AL = dL/(nL-1)*Exp(-aL*aL/2);
		double CL = Sqrt(Pi()/2)*Erf(aL/Sqrt(2));
		double CR = Sqrt(Pi()/2)*Erf(aR/Sqrt(2));
		if(y < sigma*N*AL){//below lower boundary, use inverted power law CDF (left side)
			double BL = dL/(nL-1)*Exp(-aL*aL/2);
			x = mu + sigma*(-dL*Power(y/(sigma*N*BL),1/(-nL+1)) - aL + dL);
		}
		else if(y > sigma*N*(AL+CL+CR)){//above lower boundary, use inverted power law CDF (right side)
			double AR = dR/(nR-1)*Exp(-aR*aR/2);
			double BR = dR/(1-nR)*Exp(-aR*aR/2);
			double D = (y/(sigma*N)-AL-CL-CR-AR)/BR;
			x = mu + sigma*(dR*Power(D,1/(-nR+1)) + aR - dR);
		}
		else{//between boundaries, use gaussian CDF with proper normalization (in terms of erfc)		
			double D = 1 - Sqrt(2/Pi())*(y/(sigma*N)-AL-CL);
			x = mu + sigma*Sqrt(2)*ErfcInverse(D);
		}
	}
	
	return x;
	
}